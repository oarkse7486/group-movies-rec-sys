"""
main.py
-------
FastAPI inference server for Group Rec.

Runs model inference only - no training happens here.
At startup, model artifacts are downloaded from Hugging Face Hub if they
are not already present locally. This allows deployment on Railway and
other cloud platforms without committing large binary files to GitHub.

Models hosted at: https://huggingface.co/oarkse7486/group-rec-models

SVD is excluded from deployment (685MB, too large) - the app uses NCF
for all recommendations. Popularity model is used for candidate generation
and as a fallback when NCF cannot score a user or movie.

Endpoints:
    GET  /health                 - health check
    GET  /movies/popular         - top movies for rating UI
    POST /recommend              - generate group recommendations

Usage:
    uvicorn main:app --reload --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

from scripts.model import PopularityRecommender, SVDRecommender, NCFRecommender
from scripts.group_aggregation import (
    compute_fairness_score,
    get_strategy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hugging Face model config
# ---------------------------------------------------------------------------

HF_REPO_ID = "oarkse7486/group-rec-models"

# Maps filename on Hugging Face to local path
HF_MODEL_FILES = {
    "popularity_baseline.pkl": "models/popularity_baseline.pkl",
    "ncf_model.pt": "models/ncf_model.pt",
}


def download_models_if_missing() -> None:
    """
    Download model artifacts from Hugging Face Hub if not present locally.

    Skips download for any file that already exists locally, so this is
    safe to call on every startup without re-downloading unnecessarily.
    Models are stored at: https://huggingface.co/oarkse7486/group-rec-models
    """
    os.makedirs("models", exist_ok=True)

    for filename, local_path in HF_MODEL_FILES.items():
        if os.path.exists(local_path):
            logger.info(f"Model already present locally, skipping download: {local_path}")
            continue

        logger.info(f"Downloading {filename} from Hugging Face...")
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir="models",
            )
            logger.info(f"Downloaded {filename} successfully.")
        except Exception as e:
            logger.error(f"Failed to download {filename} from Hugging Face: {e}. "
                         f"Inference may fail if this model is required.")


# ---------------------------------------------------------------------------
# App state - models loaded once at startup
# ---------------------------------------------------------------------------

app_state: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Download missing model artifacts then load all components at startup.

    Each model is loaded inside its own try/except block so a single
    missing file does not crash the entire server. Missing models are
    logged as errors and the server starts in a degraded state rather
    than refusing to start at all.
    """
    # Download any missing models from Hugging Face before loading
    download_models_if_missing()

    logger.info("Loading models...")

    # Popularity baseline - required for candidate generation and fallback
    try:
        app_state["popularity"] = PopularityRecommender.load(
            "models/popularity_baseline.pkl"
        )
        logger.info("PopularityRecommender loaded.")
    except FileNotFoundError:
        logger.error("models/popularity_baseline.pkl not found. "
                     "Run python3 setup.py to train models.")
        app_state["popularity"] = None
    except Exception as e:
        logger.error(f"Failed to load PopularityRecommender: {e}")
        app_state["popularity"] = None

    # SVD - secondary recommendation model (not deployed, loads if present)
    try:
        app_state["svd"] = SVDRecommender.load("models/svd_model.pkl")
        logger.info("SVDRecommender loaded.")
    except FileNotFoundError:
        logger.warning("models/svd_model.pkl not found. "
                       "SVD is not deployed - this is expected in production.")
        app_state["svd"] = None
    except Exception as e:
        logger.error(f"Failed to load SVDRecommender: {e}")
        app_state["svd"] = None

    # NCF - primary recommendation model
    try:
        app_state["ncf"] = NCFRecommender.load("models/ncf_model.pt")
        logger.info("NCFRecommender loaded.")
    except FileNotFoundError:
        logger.error("models/ncf_model.pt not found. "
                     "Run python3 train_ncf.py to train the NCF model.")
        app_state["ncf"] = None
    except Exception as e:
        logger.error(f"Failed to load NCFRecommender: {e}")
        app_state["ncf"] = None

    # Movie metadata - required for building recommendation response cards
    try:
        app_state["movie_meta"] = pd.read_csv("data/processed/movie_meta.csv")
        logger.info("Movie metadata loaded.")
    except FileNotFoundError:
        logger.error("data/processed/movie_meta.csv not found. "
                     "Run python3 setup.py to generate processed data.")
        app_state["movie_meta"] = None
    except Exception as e:
        logger.error(f"Failed to load movie metadata: {e}")
        app_state["movie_meta"] = None

    loaded = [k for k, v in app_state.items() if v is not None]
    missing = [k for k, v in app_state.items() if v is None]

    if missing:
        logger.warning(f"Server started with missing components: {missing}. "
                       f"Some endpoints may return errors.")
    else:
        logger.info("All components loaded. Server ready.")

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
        description="Underlying RS model: 'ncf' or 'popularity'"
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

def _require_component(name: str):
    """
    Raise HTTP 503 if a required app component is not loaded.

    Args:
        name: key in app_state to check

    Raises:
        HTTPException 503 if the component is None
    """
    if app_state.get(name) is None:
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: '{name}' model not loaded. "
                   f"Run python3 setup.py to generate required files."
        )


def _get_model(model_name: str):
    """
    Return the requested model from app state.

    Args:
        model_name: one of 'ncf', 'svd', or 'popularity'

    Raises:
        HTTPException 400 if model_name is unknown
        HTTPException 503 if the requested model failed to load at startup
    """
    available_models = {
        "popularity": app_state.get("popularity"),
        "svd": app_state.get("svd"),
        "ncf": app_state.get("ncf"),
    }
    if model_name not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Choose from: {list(available_models)}"
        )
    if available_models[model_name] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' is not available. "
                   f"Check server logs for loading errors."
        )
    return available_models[model_name]


def _get_candidate_movies(top_n: int = 500) -> List[int]:
    """
    Return a candidate set of movie IDs for scoring.

    Scores the top top_n most popular movies rather than all 62k to keep
    inference latency acceptable in a demo setting.

    Args:
        top_n: number of candidate movies to return

    Returns:
        List of movie IDs
    """
    _require_component("popularity")
    try:
        return app_state["popularity"].get_top_movies(top_n)
    except Exception as e:
        logger.error(f"Failed to get candidate movies: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve candidate movies from popularity model."
        )


def _build_member_score_map(
    member: MemberRatings,
    model,
    candidate_ids: List[int],
) -> Dict[int, float]:
    """
    Generate predicted ratings for a member across all candidate movies.

    Uses NCF's predict_new_user() if available, which builds a pseudo user
    embedding from the member's rated movies. Falls back to the popularity
    model if NCF scoring fails.

    Args:
        member:        MemberRatings with member_id and ratings dict
        model:         loaded recommender model
        candidate_ids: list of candidate movie IDs to score

    Returns:
        Dict mapping movie_id -> predicted score in [0.5, 5.0]
    """
    # NCF path: use item embedding weighted average for new users
    if hasattr(model, "predict_new_user"):
        try:
            scores = model.predict_new_user(member.ratings, candidate_ids)
            if scores:
                return scores
            logger.warning(f"predict_new_user returned empty scores for "
                           f"member {member.member_id}, falling back to popularity.")
        except Exception as e:
            logger.error(f"predict_new_user failed for member "
                         f"{member.member_id}: {e}. Falling back to popularity.")

    # Popularity fallback
    try:
        if app_state.get("popularity") is not None:
            return app_state["popularity"].predict(None, candidate_ids)
    except Exception as e:
        logger.error(f"Popularity fallback also failed: {e}")

    # Last resort: return neutral scores for all candidates
    logger.error(f"All scoring methods failed for member {member.member_id}. "
                 f"Returning neutral scores of 3.0.")
    return {mid: 3.0 for mid in candidate_ids}


def _movie_id_to_card(movie_id: int) -> Optional[MovieCard]:
    """
    Look up movie metadata by ID.

    Args:
        movie_id: MovieLens movie ID

    Returns:
        MovieCard if found, None if not in metadata
    """
    if app_state.get("movie_meta") is None:
        return None
    try:
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
    except Exception as e:
        logger.error(f"Failed to build MovieCard for movie_id {movie_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """
    API health check.

    Returns loaded components and any missing ones so callers can detect
    degraded state without triggering a recommendation request.
    """
    loaded = [k for k, v in app_state.items() if v is not None]
    missing = [k for k, v in app_state.items() if v is None]
    status = "ok" if not missing else "degraded"
    return {
        "status": status,
        "loaded": loaded,
        "missing": missing,
    }


@app.get("/movies/popular", response_model=List[MovieCard])
def get_popular_movies(limit: int = 50):
    """
    Return popular movies for the rating UI.

    Used on the member preference screen so users can rate from a
    curated list of well-known films.

    Args:
        limit: number of movies to return (default 50)
    """
    _require_component("popularity")
    _require_component("movie_meta")

    try:
        top_ids = app_state["popularity"].get_top_movies(limit)
        cards = [_movie_id_to_card(mid) for mid in top_ids]
        result = [c for c in cards if c is not None]
        if not result:
            raise HTTPException(
                status_code=500,
                detail="No movies could be retrieved from metadata."
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_popular_movies failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve popular movies."
        )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Generate group movie recommendations.

    For each group member, predicts ratings across the candidate movie set,
    then aggregates using the chosen strategy.

    Args:
        request: RecommendRequest with members, strategy, alpha, top_k, model
    """
    _require_component("movie_meta")

    if len(request.members) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 members.")

    # Validate strategy before doing any scoring
    try:
        strategy_fn = get_strategy(request.strategy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        model = _get_model(request.model)
    except HTTPException:
        raise

    try:
        candidate_ids = _get_candidate_movies(top_n=500)
    except HTTPException:
        raise

    # Step 1: get individual predicted scores for each member
    group_scores: Dict[str, Dict[int, float]] = {}
    for member in request.members:
        try:
            scores = _build_member_score_map(member, model, candidate_ids)
            group_scores[member.member_id] = scores
        except Exception as e:
            logger.error(f"Scoring failed for member {member.member_id}: {e}. "
                         f"Skipping this member.")

    if len(group_scores) < 2:
        raise HTTPException(
            status_code=422,
            detail="Could not generate scores for at least 2 group members. "
                   "Check that members have rated at least one known movie."
        )

    # Step 2: aggregate using chosen strategy
    try:
        if request.strategy == "fairness_aware":
            ranked = strategy_fn(group_scores, alpha=request.alpha, top_k=request.top_k)
        else:
            ranked = strategy_fn(group_scores, top_k=request.top_k)
    except Exception as e:
        logger.error(f"Aggregation failed with strategy '{request.strategy}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Aggregation failed: {e}"
        )

    if not ranked:
        raise HTTPException(
            status_code=500,
            detail="Aggregation returned no recommendations. "
                   "Check that candidate movies exist in the model."
        )

    # Step 3: compute fairness summary
    try:
        fairness = compute_fairness_score(group_scores, ranked)
    except Exception as e:
        logger.error(f"Fairness computation failed: {e}")
        # Non-fatal - return neutral fairness values rather than failing
        fairness = {
            "avg_group_satisfaction": 0.0,
            "fairness_score": 0.0,
            "per_member_satisfaction": {uid: 0.0 for uid in group_scores},
        }

    # Step 4: build response
    recommendations = []
    for movie_id, group_score in ranked:
        card = _movie_id_to_card(movie_id)
        if card is None:
            continue
        member_scores = [
            MemberScore(
                member_id=uid,
                predicted_rating=round(
                    float(group_scores[uid].get(movie_id, 0.0)), 2
                )
            )
            for uid in group_scores
        ]
        recommendations.append(RecommendedMovie(
            movie=card,
            group_score=round(float(group_score), 3),
            member_scores=member_scores,
        ))

    if not recommendations:
        raise HTTPException(
            status_code=500,
            detail="No recommendations could be built. "
                   "Movie metadata may be missing for all candidate movies."
        )

    return RecommendResponse(
        recommendations=recommendations,
        fairness_summary=FairnessSummary(
            avg_group_satisfaction=round(fairness["avg_group_satisfaction"], 3),
            fairness_score=round(fairness["fairness_score"], 3),
            per_member_satisfaction={
                k: round(float(v), 3)
                for k, v in fairness["per_member_satisfaction"].items()
            },
        ),
        strategy_used=request.strategy,
        model_used=request.model,
        group_size=len(request.members),
    )
