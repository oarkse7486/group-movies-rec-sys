"""
scripts/build_features.py
-------------------------
Preprocessing pipeline for Group Rec.

Steps:
  1. Load raw MovieLens 25M ratings and movies
  2. Filter sparse users and movies (cold-start exclusion)
  3. Temporal train/test split (uses timestamps - no data leakage)
  4. Build user-item interaction matrix
  5. Save processed artifacts to data/processed/

Rationale for each step is documented inline.

Usage:
    python scripts/build_features.py
"""

import os
import pickle

import numpy as np
import pandas as pd

RAW_DIR = "data/raw/ml-25m"
PROCESSED_DIR = "data/processed"

# Filtering thresholds
MIN_USER_RATINGS = 20   # exclude users with fewer ratings (cold start)
MIN_MOVIE_RATINGS = 50  # exclude movies with fewer ratings (long tail noise)

# Train/test split: last 20% of each user's ratings by timestamp go to test
TEST_FRACTION = 0.2


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw ratings and movies CSVs from data/raw/."""
    print("  Loading raw data...")
    ratings = pd.read_csv(os.path.join(RAW_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
    links = pd.read_csv(os.path.join(RAW_DIR, "links.csv"))

    # Attach TMDB id to movies for poster fetching in the app
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    print(f"  Ratings: {len(ratings):,} | Movies: {len(movies):,}")
    return ratings, movies


def filter_sparse_entities(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Remove users with fewer than MIN_USER_RATINGS and movies with fewer than MIN_MOVIE_RATINGS.

    Rationale: cold-start users and extremely obscure movies degrade model
    quality and inflate RMSE without meaningful signal. This is standard
    practice in RS literature (Ricci et al., 2015).
    """
    print(f"  Filtering: min {MIN_USER_RATINGS} ratings/user, "
          f"{MIN_MOVIE_RATINGS} ratings/movie...")

    user_counts = ratings["userId"].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    ratings = ratings[ratings["userId"].isin(valid_users)]

    movie_counts = ratings["movieId"].value_counts()
    valid_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
    ratings = ratings[ratings["movieId"].isin(valid_movies)]

    print(f"  After filter: {len(ratings):,} ratings | "
          f"{ratings['userId'].nunique():,} users | "
          f"{ratings['movieId'].nunique():,} movies")
    return ratings.reset_index(drop=True)


def temporal_train_test_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train/test using timestamps.

    For each user, the most recent TEST_FRACTION of their ratings go to the
    test set. All earlier ratings go to train.

    Rationale: random splitting leaks future data into training (a user's
    future ratings influence predictions of their past). Temporal splitting
    simulates a realistic deployment scenario where we predict what a user
    will rate next given their history so far.
    """
    print("  Performing temporal train/test split...")
    ratings = ratings.sort_values(["userId", "timestamp"])

    def split_user(group: pd.DataFrame) -> pd.Series:
        n_test = max(1, int(len(group) * TEST_FRACTION))
        labels = ["train"] * (len(group) - n_test) + ["test"] * n_test
        return pd.Series(labels, index=group.index)

    split_labels = ratings.groupby("userId", group_keys=False).apply(split_user)
    train = ratings[split_labels == "train"].copy()
    test = ratings[split_labels == "test"].copy()

    print(f"  Train: {len(train):,} ratings | Test: {len(test):,} ratings")
    return train, test


def build_movie_index(movies: pd.DataFrame, valid_movie_ids: pd.Index) -> pd.DataFrame:
    """
    Build a clean movie metadata table with reindexed integer IDs.

    Returns a DataFrame with columns: movieId, title, genres, tmdbId, movie_idx
    """
    movie_meta = movies[movies["movieId"].isin(valid_movie_ids)].copy()
    movie_meta = movie_meta.reset_index(drop=True)
    movie_meta["movie_idx"] = movie_meta.index
    return movie_meta


def build_user_index(train: pd.DataFrame) -> pd.DataFrame:
    """Build a user ID to integer index mapping."""
    user_ids = train["userId"].unique()
    user_map = pd.DataFrame({
        "userId": user_ids,
        "user_idx": range(len(user_ids))
    })
    return user_map


def run_preprocessing_pipeline() -> None:
    """
    Execute the full preprocessing pipeline and save artifacts.

    Saved artifacts:
        data/processed/train.csv
        data/processed/test.csv
        data/processed/movie_meta.csv
        data/processed/user_map.pkl
        data/processed/movie_map.pkl
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ratings, movies = load_raw_data()
    ratings = filter_sparse_entities(ratings)
    train, test = temporal_train_test_split(ratings)

    movie_meta = build_movie_index(movies, train["movieId"].unique())
    user_map = build_user_index(train)

    movie_id_to_idx = dict(zip(movie_meta["movieId"], movie_meta["movie_idx"]))
    user_id_to_idx = dict(zip(user_map["userId"], user_map["user_idx"]))

    train["movie_idx"] = train["movieId"].map(movie_id_to_idx)
    train["user_idx"] = train["userId"].map(user_id_to_idx)
    test["movie_idx"] = test["movieId"].map(movie_id_to_idx)
    test["user_idx"] = test["userId"].map(user_id_to_idx)

    # Drop test rows for unseen movies/users
    test = test.dropna(subset=["movie_idx", "user_idx"])

    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    movie_meta.to_csv(os.path.join(PROCESSED_DIR, "movie_meta.csv"), index=False)

    with open(os.path.join(PROCESSED_DIR, "user_map.pkl"), "wb") as f:
        pickle.dump(user_id_to_idx, f)
    with open(os.path.join(PROCESSED_DIR, "movie_map.pkl"), "wb") as f:
        pickle.dump(movie_id_to_idx, f)

    n_users = train["userId"].nunique()
    n_movies = train["movieId"].nunique()
    print(f"  Saved processed data: {n_users:,} users, {n_movies:,} movies")


if __name__ == "__main__":
    run_preprocessing_pipeline()
