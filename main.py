"""
main.py
-------
FastAPI inference server for Group Rec.

Runs model inference only - no training happens here.
All models are loaded once at startup from the models/ directory.

Endpoints:
    GET  /health                 - health check
    GET  /movies/popular         - top movies for rating UI
    POST /users/profile          - build taste profile from explicit ratings
    POST /recommend              - generate group recommendations

Usage:
    uvicorn main:app --reload --port 8000
"""

import os
import pickle
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scripts.model import PopularityRecommender, SVDRecommender, NCFRecommender
from scripts.group_aggregation import (
    least_misery,
    average_satisfaction,
    fairness_aware,
    compute_fairness_score,
    get_strategy,
)

# ---------------------------------------------------------------------------
# App state - models loaded once at startup
# ---------------------------------------------------------------------------

app_state: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts and movie metadata at startup."""
    print("Loading models...")

    app_state["popularity"] = PopularityRecommender.load(
        "models/popularity_baseline.pkl"
    )
    app_state["svd"] = SVDRecommender.load("models/svd_model.pkl")
    app_state["ncf"] = NCFRecommender.load("models/ncf_model.pt")
    app_state["movie_meta"] = pd.read_csv("data/processed/movie_meta.csv")

    print("All models loaded. Server ready.")
    yield
    app_state.clear()


app = FastAPI(
    title="Group Rec API",
    description="Group-aware movie recommendation system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MemberRatings(BaseModel):
    """Explicit ratings provided by one group member."""
    member_id: str = Field(..., description="Unique identifier for this member")
    ratings: Dict[int, float] = Field(
        ..., description="Dict of {movie_id: rating (0.5-5.0)}"
    )


class RecommendRequest(BaseModel):
    """Request body for /recommend."""
    members: List[MemberRatings] = Field(
        ..., min_length=2, max_length=10,
        description="List of group member profiles (2-10 members)"
    )
    strategy: str = Field(
        default="fairness_aware",
        description="Aggregation strategy: 'least_misery', 'average', 'fairness_aware'"
    )
    alpha: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Fairness-aware blend weight (only used if strategy=fairness_aware)"
    )
    top_k: int = Field(default=10, ge=1, le=20)
    model: str = Field(
        default="ncf",
        description="Underlying RS model: 'popularity', 'svd', 'ncf'"
    )


class MovieCard(BaseModel):
    """Movie metadata for UI rendering."""
    movie_id: int
    title: str
    genres: List[str]
    tmdb_id: Optional[int]


class MemberScore(BaseModel):
    """Per-member predicted satisfaction for one movie."""
    member_id: str
    predicted_rating: float


class RecommendedMovie(BaseModel):
    """One recommended movie with group and per-member scores."""
    movie: MovieCard
    group_score: float
    member_scores: List[MemberScore]


class FairnessSummary(BaseModel):
    """Group-level fairness metrics."""
    avg_group_satisfaction: float
    fairness_score: float
    per_member_satisfaction: Dict[str, float]


class RecommendResponse(BaseModel):
    """Full recommendation response."""
    recommendations: List[RecommendedMovie]
    fairness_summary: FairnessSummary
    strategy_used: str
    model_used: str
    group_size: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model(model_name: str):
    """Return the requested model from app state."""
    models = {
        "popularity": app_state["popularity"],
        "svd": app_state["svd"],
        "ncf": app_state["ncf"],
    }
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Choose from: {list(models)}"
        )
    return models[model_name]


def _get_candidate_movies(top_n: int = 500) -> List[int]:
    """
    Return a candidate set of movie IDs for scoring.

    We score the top 500 most popular movies rather than all 62k to keep
    inference latency acceptable in a demo setting.
    """
    return app_state["popularity"].get_top_movies(top_n)


def _build_member_score_map(
    member: MemberRatings,
    model,
    candidate_ids: List[int],
) -> Dict[int, float]:
    """
    Generate predicted ratings for a member across all candidate movies.

    For known users (MovieLens user IDs), uses the model's .predict() method.
    For new users (custom member_ids), uses NCF's new-user embedding approach
    or SVD's average fallback.
    """
    if hasattr(model, "predict_new_user"):
        return model.predict_new_user(member.ratings, candidate_ids)

    try:
        user_id = int(member.member_id)
        return model.predict(user_id, candidate_ids)
    except (ValueError, TypeError):
        return app_state["popularity"].predict(None, candidate_ids)


def _movie_id_to_card(movie_id: int) -> Optional[MovieCard]:
    """Look up movie metadata by ID."""
    meta = app_state["movie_meta"]
    row = meta[meta["movieId"] == movie_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return MovieCard(
        movie_id=int(r["movieId"]),
        title=str(r["title"]),
        genres=str(r["genres"]).split("|"),
        tmdb_id=int(r["tmdbId"]) if pd.notna(r.get("tmdbId")) else None,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """API health check."""
    return {"status": "ok", "models_loaded": list(app_state.keys())}


@app.get("/movies/popular", response_model=List[MovieCard])
def get_popular_movies(limit: int = 50):
    """
    Return popular movies for the rating UI.

    Used on the member preference screen so users can rate from a
    curated list of well-known films.
    """
    top_ids = app_state["popularity"].get_top_movies(limit)
    cards = [_movie_id_to_card(mid) for mid in top_ids]
    return [c for c in cards if c is not None]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Generate group movie recommendations.

    For each group member, predicts ratings across the candidate movie set,
    then aggregates using the chosen strategy.
    """
    if len(request.members) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 members.")

    model = _get_model(request.model)
    candidate_ids = _get_candidate_movies(top_n=500)

    # Step 1: get individual predicted scores for each member
    group_scores: Dict[str, Dict[int, float]] = {}
    for member in request.members:
        group_scores[member.member_id] = _build_member_score_map(
            member, model, candidate_ids
        )

    # Step 2: aggregate using chosen strategy
    strategy_fn = get_strategy(request.strategy)

    if request.strategy == "fairness_aware":
        ranked = strategy_fn(group_scores, alpha=request.alpha, top_k=request.top_k)
    else:
        ranked = strategy_fn(group_scores, top_k=request.top_k)

    # Step 3: compute fairness summary
    fairness = compute_fairness_score(group_scores, ranked)

    # Step 4: build response
    recommendations = []
    for movie_id, group_score in ranked:
        card = _movie_id_to_card(movie_id)
        if card is None:
            continue
        member_scores = [
            MemberScore(
                member_id=uid,
                predicted_rating=round(group_scores[uid].get(movie_id, 0.0), 2)
            )
            for uid in group_scores
        ]
        recommendations.append(RecommendedMovie(
            movie=card,
            group_score=round(group_score, 3),
            member_scores=member_scores,
        ))

    return RecommendResponse(
        recommendations=recommendations,
        fairness_summary=FairnessSummary(
            avg_group_satisfaction=round(fairness["avg_group_satisfaction"], 3),
            fairness_score=round(fairness["fairness_score"], 3),
            per_member_satisfaction={
                k: round(v, 3)
                for k, v in fairness["per_member_satisfaction"].items()
            },
        ),
        strategy_used=request.strategy,
        model_used=request.model,
        group_size=len(request.members),
    )
