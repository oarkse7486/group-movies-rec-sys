"""
scripts/group_aggregation.py
-----------------------------
Group aggregation strategies for Group Rec.

Given a dictionary of {user_id: {movie_id: predicted_score}} for each group
member, these functions aggregate individual scores into a single ranked list
of group recommendations.

Three strategies:
  1. Least Misery      - group score = min of individual scores
  2. Average           - group score = mean of individual scores
  3. Fairness-Aware    - group score = alpha*avg + (1-alpha)*least_misery

Reference:
  Masthoff, J. (2011). Group Recommender Systems: Combining Individual Models.
  In Recommender Systems Handbook. Springer.
"""

from typing import Dict, List, Tuple
import numpy as np


ScoreMap = Dict[int, float]          # movie_id -> predicted score
GroupScores = Dict[int, ScoreMap]    # user_id -> ScoreMap
Recommendation = Tuple[int, float]   # (movie_id, group_score)


def least_misery(
    group_scores: GroupScores,
    top_k: int = 10
) -> List[Recommendation]:
    """
    Least Misery aggregation strategy.

    The group's predicted score for a movie is the MINIMUM individual score
    across all members. Ensures no one is deeply unhappy with the recommendation,
    but can underperform when one member is an outlier.

    Args:
        group_scores: Dict mapping user_id -> {movie_id -> predicted_score}
        top_k: number of recommendations to return

    Returns:
        Sorted list of (movie_id, group_score) tuples, highest score first
    """
    all_movies = _get_common_movies(group_scores)
    aggregated = {}

    for movie_id in all_movies:
        scores = [group_scores[uid][movie_id] for uid in group_scores
                  if movie_id in group_scores[uid]]
        if scores:
            aggregated[movie_id] = min(scores)

    return _rank(aggregated, top_k)


def average_satisfaction(
    group_scores: GroupScores,
    top_k: int = 10
) -> List[Recommendation]:
    """
    Average Satisfaction aggregation strategy.

    The group's predicted score for a movie is the MEAN of individual scores.
    Maximizes overall group utility but can systematically ignore one outlier
    member.

    Args:
        group_scores: Dict mapping user_id -> {movie_id -> predicted_score}
        top_k: number of recommendations to return

    Returns:
        Sorted list of (movie_id, group_score) tuples, highest score first
    """
    all_movies = _get_common_movies(group_scores)
    aggregated = {}

    for movie_id in all_movies:
        scores = [group_scores[uid][movie_id] for uid in group_scores
                  if movie_id in group_scores[uid]]
        if scores:
            aggregated[movie_id] = float(np.mean(scores))

    return _rank(aggregated, top_k)


def fairness_aware(
    group_scores: GroupScores,
    alpha: float = 0.6,
    top_k: int = 10
) -> List[Recommendation]:
    """
    Fairness-Aware aggregation strategy.

    Blends Average Satisfaction and Least Misery:
        score = alpha * average + (1 - alpha) * least_misery

    When alpha = 1.0 this reduces to pure average satisfaction.
    When alpha = 0.0 this reduces to pure least misery.
    The default alpha = 0.6 prioritizes group happiness while still protecting
    the most dissatisfied member.

    Args:
        group_scores: Dict mapping user_id -> {movie_id -> predicted_score}
        alpha: blend weight. Higher = more average, lower = more least misery.
        top_k: number of recommendations to return

    Returns:
        Sorted list of (movie_id, group_score) tuples, highest score first
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    all_movies = _get_common_movies(group_scores)
    aggregated = {}

    for movie_id in all_movies:
        scores = [group_scores[uid][movie_id] for uid in group_scores
                  if movie_id in group_scores[uid]]
        if scores:
            avg = float(np.mean(scores))
            lm = min(scores)
            aggregated[movie_id] = alpha * avg + (1.0 - alpha) * lm

    return _rank(aggregated, top_k)


def compute_fairness_score(
    group_scores: GroupScores,
    recommendations: List[Recommendation]
) -> Dict[str, float]:
    """
    Compute group-level satisfaction and fairness metrics for a recommendation list.

    Fairness is measured as the standard deviation of per-member average
    satisfaction across the recommended items. Lower std = fairer distribution.

    Args:
        group_scores: Dict mapping user_id -> {movie_id -> predicted_score}
        recommendations: Output of any aggregation function

    Returns:
        Dict with keys:
            - avg_group_satisfaction: mean predicted score across all members and recs
            - fairness_score: 1 - normalized_std (higher = fairer, range [0, 1])
            - per_member_satisfaction: {user_id: avg score over recommended items}
    """
    rec_movie_ids = [movie_id for movie_id, _ in recommendations]
    per_member = {}

    for uid, scores in group_scores.items():
        member_scores = [scores[mid] for mid in rec_movie_ids if mid in scores]
        per_member[uid] = float(np.mean(member_scores)) if member_scores else 0.0

    satisfactions = list(per_member.values())
    avg_satisfaction = float(np.mean(satisfactions))
    std_satisfaction = float(np.std(satisfactions))

    # Normalize std to [0, 1] range (max possible std on a 1-5 scale is 2.0)
    normalized_std = std_satisfaction / 2.0
    fairness_score = max(0.0, 1.0 - normalized_std)

    return {
        "avg_group_satisfaction": avg_satisfaction,
        "fairness_score": fairness_score,
        "per_member_satisfaction": per_member,
    }


def get_strategy(name: str):
    """
    Return an aggregation function by name.

    Args:
        name: one of 'least_misery', 'average', 'fairness_aware'

    Returns:
        Callable aggregation function
    """
    strategies = {
        "least_misery": least_misery,
        "average": average_satisfaction,
        "fairness_aware": fairness_aware,
    }
    if name not in strategies:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(strategies)}")
    return strategies[name]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_common_movies(group_scores: GroupScores) -> set:
    """Return the union of all movie IDs across all group members."""
    all_movies = set()
    for scores in group_scores.values():
        all_movies.update(scores.keys())
    return all_movies


def _rank(aggregated: Dict[int, float], top_k: int) -> List[Recommendation]:
    """Sort aggregated scores descending and return top_k."""
    sorted_recs = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    return sorted_recs[:top_k]
