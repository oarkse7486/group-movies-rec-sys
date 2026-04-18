"""
experiment.py
-------------
Experiment: How does group size (2-10 members) affect the accuracy-fairness
tradeoff across the three aggregation strategies?

For each group size from 2 to 10, constructs N_GROUPS synthetic groups by
randomly sampling MovieLens users from the test set. For each group, generates
recommendations using all three aggregation strategies on top of the NCF model,
then measures average group satisfaction and fairness score.

Results are saved to data/outputs/experiment_results.csv and a summary table
is printed to the terminal.

Motivation: real-world viewing groups range from 2 to 10 people. Understanding
where each aggregation strategy degrades with group size is directly actionable
for product decisions - for example, automatically switching from Average
Satisfaction to Fairness-Aware when group size exceeds a threshold.

Usage:
    python3 experiment.py

Note: random_state is passed as a parameter with default None.
Set to a fixed integer (e.g. 42) for reproducible results to report.
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional, List, Dict

from scripts.model import NCFRecommender, PopularityRecommender
from scripts.group_aggregation import (
    least_misery,
    average_satisfaction,
    fairness_aware,
    compute_fairness_score,
)

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
OUTPUTS_DIR = "data/outputs"

# Number of synthetic groups to construct at each group size
N_GROUPS = 100

# Number of candidate movies to score per group (top popularity)
N_CANDIDATES = 200

# Top-k recommendations to evaluate per group
TOP_K = 10

# Group sizes to test
GROUP_SIZES = list(range(2, 11))

# Aggregation strategies to compare
STRATEGIES = {
    "least_misery": least_misery,
    "average_satisfaction": average_satisfaction,
    "fairness_aware": fairness_aware,
}


# ---------------------------------------------------------------------------
# Synthetic group construction
# ---------------------------------------------------------------------------

def build_synthetic_groups(
    test_df: pd.DataFrame,
    group_size: int,
    n_groups: int,
    random_state: Optional[int] = None
) -> List[List[int]]:
    """
    Construct synthetic groups by randomly sampling user IDs from the test set.

    Each group is a list of user IDs. Users are sampled without replacement
    within each group but with replacement across groups.

    Args:
        test_df:      test DataFrame with column [userId]
        group_size:   number of members per group
        n_groups:     number of groups to construct
        random_state: seed for reproducibility. None = random each run.

    Returns:
        List of groups, each group is a list of user_id integers
    """
    rng = np.random.default_rng(random_state)
    all_users = test_df["userId"].unique().tolist()

    groups = []
    for _ in range(n_groups):
        members = rng.choice(all_users, size=group_size, replace=False).tolist()
        groups.append([int(m) for m in members])
    return groups


# ---------------------------------------------------------------------------
# Group evaluation
# ---------------------------------------------------------------------------

def evaluate_group(
    group: List[int],
    ncf: NCFRecommender,
    candidate_ids: List[int],
    strategy_name: str,
    alpha: float = 0.6
) -> Dict[str, float]:
    """
    Generate recommendations for one synthetic group and compute metrics.

    Args:
        group:         list of user IDs in the group
        ncf:           trained NCFRecommender
        candidate_ids: list of candidate movie IDs to score
        strategy_name: one of 'least_misery', 'average_satisfaction', 'fairness_aware'
        alpha:         blend weight for fairness_aware strategy

    Returns:
        Dict with keys: avg_group_satisfaction, fairness_score
    """
    # Get individual predicted scores for each member
    group_scores = {}
    for user_id in group:
        scores = ncf.predict(user_id, candidate_ids)
        if scores:
            group_scores[str(user_id)] = scores

    if len(group_scores) < 2:
        return None

    # Aggregate using chosen strategy
    strategy_fn = STRATEGIES[strategy_name]
    if strategy_name == "fairness_aware":
        ranked = strategy_fn(group_scores, alpha=alpha, top_k=TOP_K)
    else:
        ranked = strategy_fn(group_scores, top_k=TOP_K)

    if not ranked:
        return None

    # Compute metrics
    metrics = compute_fairness_score(group_scores, ranked)
    return {
        "avg_group_satisfaction": metrics["avg_group_satisfaction"],
        "fairness_score": metrics["fairness_score"],
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Run the full group size vs accuracy-fairness experiment.

    For each group size in GROUP_SIZES and each strategy in STRATEGIES,
    constructs N_GROUPS synthetic groups and averages metrics across them.

    Args:
        random_state: seed for reproducible group sampling. None = random.

    Returns:
        DataFrame with columns:
            group_size, strategy, avg_group_satisfaction, fairness_score
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Loading models and data...")
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    ncf = NCFRecommender.load(os.path.join(MODELS_DIR, "ncf_model.pt"))
    popularity = PopularityRecommender.load(
        os.path.join(MODELS_DIR, "popularity_baseline.pkl")
    )

    # Use top N_CANDIDATES most popular movies as candidate set
    candidate_ids = popularity.get_top_movies(N_CANDIDATES)

    print(f"Running experiment: {len(GROUP_SIZES)} group sizes x "
          f"{len(STRATEGIES)} strategies x {N_GROUPS} groups each")
    if random_state is not None:
        print(f"random_state={random_state} (reproducible)")
    else:
        print("random_state=None (random)")

    rows = []

    for group_size in GROUP_SIZES:
        print(f"\n  Group size {group_size}...")
        groups = build_synthetic_groups(
            test_df, group_size, N_GROUPS, random_state=random_state
        )

        for strategy_name in STRATEGIES:
            satisfactions = []
            fairness_scores = []

            for group in groups:
                result = evaluate_group(
                    group, ncf, candidate_ids, strategy_name
                )
                if result is None:
                    continue
                satisfactions.append(result["avg_group_satisfaction"])
                fairness_scores.append(result["fairness_score"])

            avg_sat = float(np.mean(satisfactions)) if satisfactions else 0.0
            avg_fair = float(np.mean(fairness_scores)) if fairness_scores else 0.0

            print(f"    {strategy_name:<25} | "
                  f"satisfaction={avg_sat:.4f} | fairness={avg_fair:.4f}")

            rows.append({
                "group_size": group_size,
                "strategy": strategy_name,
                "avg_group_satisfaction": round(avg_sat, 4),
                "fairness_score": round(avg_fair, 4),
            })

    return pd.DataFrame(rows)


def print_summary(results_df: pd.DataFrame) -> None:
    """
    Print a formatted summary table of experiment results.

    Args:
        results_df: DataFrame returned by run_experiment()
    """
    print("\n" + "=" * 75)
    print("  Experiment Results: Group Size vs Accuracy-Fairness Tradeoff")
    print("=" * 75)
    print(f"  {'Group Size':<12} {'Strategy':<26} {'Avg Satisfaction':>18} {'Fairness Score':>15}")
    print("  " + "-" * 73)

    for _, row in results_df.iterrows():
        print(f"  {int(row['group_size']):<12} {row['strategy']:<26} "
              f"{row['avg_group_satisfaction']:>18.4f} {row['fairness_score']:>15.4f}")

    print("=" * 75)


def main(random_state: Optional[int] = None) -> None:
    """
    Run experiment, print summary, and save results to CSV.

    Args:
        random_state: seed for reproducible group sampling.
                      None = random each run (default).
                      Set to a fixed integer when reporting results in the paper.
                      The specific value chosen is arbitrary; any fixed integer
                      produces equally reproducible results.
    """
    results_df = run_experiment(random_state=random_state)

    csv_path = os.path.join(OUTPUTS_DIR, "experiment_results.csv")
    results_df.to_csv(csv_path, index=False)

    print_summary(results_df)
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    main(random_state=42)
