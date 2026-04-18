"""
evaluate.py
-----------
Evaluates all three Group Rec models on the held-out test set and saves
results to data/outputs/evaluation_results.csv.

Metrics computed for each model:
  - RMSE    (Root Mean Squared Error)       - standard rating prediction accuracy
  - MAE     (Mean Absolute Error)           - less sensitive to outliers than RMSE
  - NDCG@10 (Normalized Discounted         - ranking quality: are good movies
             Cumulative Gain at 10)           appearing at the top of the list?

RMSE and MAE measure how accurately each model predicts a rating value.
NDCG@10 measures whether the model ranks good movies above bad ones for
each user, which is more directly relevant to recommendation quality.

Acknowledgement: no validation set was used during training. The 80/20
temporal train/test split means hyperparameters were not tuned on a
separate val set. This is noted as a limitation in the report.

random_state parameter:
  Pass an integer for reproducible sampling (same users and rows every run).
  Pass None for a fresh random sample each run.
  Default is None (random). Set to a fixed integer like 42 when you need
  to share exact results with others or include them in a report.
  The value 42 has no mathematical significance - it is a community
  convention borrowed from "The Hitchhiker's Guide to the Galaxy" and is
  used purely to signal reproducible randomness.

Usage:
    python3 evaluate.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from scripts.model import PopularityRecommender, SVDRecommender, NCFRecommender

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
OUTPUTS_DIR = "data/outputs"

TOP_K = 10
NDCG_SAMPLE_USERS = 500 # try to increase to 5,000 bc total test set is 4 million
RMSE_SAMPLE_SIZE = 10000


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_rmse(predictions: List[float], actuals: List[float]) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: list of predicted ratings
        actuals:     list of actual ratings

    Returns:
        RMSE as a float
    """
    errors = [(p - a) ** 2 for p, a in zip(predictions, actuals)]
    return float(np.sqrt(np.mean(errors)))


def compute_mae(predictions: List[float], actuals: List[float]) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: list of predicted ratings
        actuals:     list of actual ratings

    Returns:
        MAE as a float
    """
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    return float(np.mean(errors))


def compute_ndcg_at_k(
    user_predictions: Dict[int, float],
    user_actuals: Dict[int, float],
    k: int = 10
) -> float:
    """
    Compute NDCG@k for a single user.

    Ranks movies by predicted score, then measures how well that ranking
    aligns with actual ratings using the DCG formula with log2 discounting.

    A movie is considered "relevant" if its actual rating >= 4.0.

    Only evaluates on movies the user actually rated in the test set.
    Unrated movies are skipped entirely because "not watched" does not mean
    "disliked" - treating missing ratings as 0 would introduce exposure bias
    and make the metric meaningless.

    Args:
        user_predictions: Dict of {movie_id: predicted_score}
        user_actuals:     Dict of {movie_id: actual_rating}
        k:                number of top recommendations to evaluate

    Returns:
        NDCG@k score in [0, 1] for this user
    """
    ranked = sorted(user_predictions.items(), key=lambda x: x[1], reverse=True)[:k]

    dcg = 0.0
    for rank, (movie_id, _) in enumerate(ranked, start=1):
        actual = user_actuals.get(movie_id, None)
        if actual is None:
            continue  # skip unrated movies - "not watched" != "disliked" (exposure bias)
        relevance = 1.0 if actual >= 4.0 else 0.0
        dcg += relevance / np.log2(rank + 1)

    ideal_relevances = sorted(
        [1.0 if r >= 4.0 else 0.0 for r in user_actuals.values()],
        reverse=True
    )[:k]
    idcg = sum(
        rel / np.log2(rank + 1)
        for rank, rel in enumerate(ideal_relevances, start=1)
    )

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Per-model evaluation functions
# ---------------------------------------------------------------------------

def evaluate_popularity(
    model: PopularityRecommender,
    test_df: pd.DataFrame,
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate PopularityRecommender on the test set.

    Args:
        model:        trained PopularityRecommender
        test_df:      test DataFrame with columns [userId, movieId, rating]
        random_state: seed for reproducible sampling. None = random each run.

    Returns:
        Dict with keys: model, rmse, mae, ndcg_at_10
    """
    print("  Evaluating Popularity Baseline...")
    sample = test_df.sample(min(RMSE_SAMPLE_SIZE, len(test_df)),
                            random_state=random_state)

    predictions, actuals = [], []
    for _, row in sample.iterrows():
        pred = model.predict(None, [int(row["movieId"])])
        predictions.append(pred.get(int(row["movieId"]), 3.0))
        actuals.append(row["rating"])

    rmse = compute_rmse(predictions, actuals)
    mae = compute_mae(predictions, actuals)
    ndcg = _compute_ndcg_for_model(model, test_df,
                                   use_user_id=False,
                                   random_state=random_state)

    return {"model": "Popularity Baseline", "rmse": rmse,
            "mae": mae, "ndcg_at_10": ndcg}


def evaluate_svd(
    model: SVDRecommender,
    test_df: pd.DataFrame,
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate SVDRecommender on the test set.

    Args:
        model:        trained SVDRecommender
        test_df:      test DataFrame with columns [userId, movieId, rating]
        random_state: seed for reproducible sampling. None = random each run.

    Returns:
        Dict with keys: model, rmse, mae, ndcg_at_10
    """
    print("  Evaluating SVD...")
    sample = test_df.sample(min(RMSE_SAMPLE_SIZE, len(test_df)),
                            random_state=random_state)

    predictions, actuals = [], []
    for _, row in sample.iterrows():
        pred = model.predict(int(row["userId"]), [int(row["movieId"])])
        predictions.append(pred.get(int(row["movieId"]), 3.0))
        actuals.append(row["rating"])

    rmse = compute_rmse(predictions, actuals)
    mae = compute_mae(predictions, actuals)
    ndcg = _compute_ndcg_for_model(model, test_df,
                                   use_user_id=True,
                                   random_state=random_state)

    return {"model": "SVD", "rmse": rmse, "mae": mae, "ndcg_at_10": ndcg}


def evaluate_ncf(
    model: NCFRecommender,
    test_df: pd.DataFrame,
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate NCFRecommender on the test set.

    Args:
        model:        trained NCFRecommender
        test_df:      test DataFrame with columns [userId, movieId, rating]
        random_state: seed for reproducible sampling. None = random each run.

    Returns:
        Dict with keys: model, rmse, mae, ndcg_at_10
    """
    print("  Evaluating NCF...")
    sample = test_df.sample(min(RMSE_SAMPLE_SIZE, len(test_df)),
                            random_state=random_state)

    predictions, actuals = [], []
    for _, row in sample.iterrows():
        pred = model.predict(int(row["userId"]), [int(row["movieId"])])
        predictions.append(pred.get(int(row["movieId"]), 3.0))
        actuals.append(row["rating"])

    rmse = compute_rmse(predictions, actuals)
    mae = compute_mae(predictions, actuals)
    ndcg = _compute_ndcg_for_model(model, test_df,
                                   use_user_id=True,
                                   random_state=random_state)

    return {"model": "NCF", "rmse": rmse, "mae": mae, "ndcg_at_10": ndcg}


def _compute_ndcg_for_model(
    model,
    test_df: pd.DataFrame,
    use_user_id: bool = True,
    random_state: Optional[int] = None
) -> float:
    """
    Compute average NDCG@10 across a sample of test users.

    Args:
        model:        any trained recommender with a .predict() method
        test_df:      test DataFrame
        use_user_id:  if True, pass user_id to predict(); False for popularity
        random_state: seed for reproducible user sampling. None = random.

    Returns:
        Average NDCG@10 across sampled users
    """
    sampled_users = (
        test_df["userId"]
        .drop_duplicates()
        .sample(min(NDCG_SAMPLE_USERS, test_df["userId"].nunique()),
                random_state=random_state)
        .tolist()
    )

    ndcg_scores = []

    for user_id in sampled_users:
        user_test = test_df[test_df["userId"] == user_id]
        if len(user_test) < 5:
            continue

        user_actuals = dict(zip(user_test["movieId"], user_test["rating"]))
        uid = int(user_id) if use_user_id else None
        preds = model.predict(uid, [int(m) for m in user_actuals.keys()])

        if not preds:
            continue

        score = compute_ndcg_at_k(preds, user_actuals, k=TOP_K)
        ndcg_scores.append(score)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(random_state: Optional[int] = None) -> None:
    """
    Load all models, evaluate on test set, print and save results.

    Args:
        random_state: seed for reproducible sampling across all evaluations.
                      None = random each run (default).
                      Set to a fixed integer (e.g. 42) when reporting final
                      results in a paper or sharing with others.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    print(f"  Test set: {len(test_df):,} ratings, "
          f"{test_df['userId'].nunique():,} users")

    if random_state is not None:
        print(f"  random_state={random_state} (reproducible sampling)")
    else:
        print("  random_state=None (random sampling)")

    print("\nLoading models...")
    popularity = PopularityRecommender.load(
        os.path.join(MODELS_DIR, "popularity_baseline.pkl")
    )
    svd = SVDRecommender.load(os.path.join(MODELS_DIR, "svd_model.pkl"))
    ncf = NCFRecommender.load(os.path.join(MODELS_DIR, "ncf_model.pt"))

    print("\nRunning evaluation...")
    results = [
        evaluate_popularity(popularity, test_df, random_state=random_state),
        evaluate_svd(svd, test_df, random_state=random_state),
        evaluate_ncf(ncf, test_df, random_state=random_state),
    ]

    results_df = pd.DataFrame(results).round(4)
    csv_path = os.path.join(OUTPUTS_DIR, "evaluation_results.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 55)
    print("  Evaluation Results")
    print("=" * 55)
    print(f"  {'Model':<25} {'RMSE':>8} {'MAE':>8} {'NDCG@10':>10}")
    print("  " + "-" * 53)
    for row in results:
        print(f"  {row['model']:<25} {row['rmse']:>8.4f} "
              f"{row['mae']:>8.4f} {row['ndcg_at_10']:>10.4f}")
    print("=" * 55)
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    # Set random_state to an integer for reproducible results to include
    # in your report, or leave as None for a fresh random sample each run.
    main(random_state=42)
