"""
Ranking-level bias metrics for the ranking impact study (Experiment 9).

These metrics translate score-level TCD into ranking-level consequences,
addressing the reviewer concern: "Score changed by 3%, but does ranking
actually change?"

Metrics:
  - Rank flip rate: % of pairs where identity swap changes relative order
  - Top-k exposure disparity: difference in inclusion rate at top-k
  - nDCG change: normalised DCG shift under identity swap
  - MRR change: Mean Reciprocal Rank shift
  - Exposure ratio: position-weighted exposure for group A vs B
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def rank_flip_rate(
    scores_original: np.ndarray,
    scores_swapped: np.ndarray,
    target_idx: int,
) -> float:
    """Fraction of candidate pools where identity swap changes target rank.

    Args:
        scores_original: Scores for all documents in pool (including target).
        scores_swapped: Scores after swapping identity marker in target doc.
        target_idx: Index of the target document in the pool.

    Returns:
        1.0 if rank changed, 0.0 otherwise.
    """
    rank_orig = int((scores_original > scores_original[target_idx]).sum()) + 1
    rank_swap = int((scores_swapped > scores_swapped[target_idx]).sum()) + 1
    return 1.0 if rank_orig != rank_swap else 0.0


def rank_displacement(
    scores_original: np.ndarray,
    scores_swapped: np.ndarray,
    target_idx: int,
) -> int:
    """Signed rank displacement of target document after identity swap.

    Positive = moved down (worse), negative = moved up (better).
    """
    rank_orig = int((scores_original > scores_original[target_idx]).sum()) + 1
    rank_swap = int((scores_swapped > scores_swapped[target_idx]).sum()) + 1
    return rank_swap - rank_orig


def top_k_inclusion_change(
    scores_original: np.ndarray,
    scores_swapped: np.ndarray,
    target_idx: int,
    k: int = 10,
) -> int:
    """Whether identity swap causes target to enter/leave top-k.

    Returns:
        +1 if target enters top-k after swap
        -1 if target leaves top-k after swap
         0 if no change in top-k membership
    """
    in_topk_orig = scores_original[target_idx] >= np.partition(
        scores_original, -k
    )[-k]
    in_topk_swap = scores_swapped[target_idx] >= np.partition(
        scores_swapped, -k
    )[-k]

    if not in_topk_orig and in_topk_swap:
        return +1
    elif in_topk_orig and not in_topk_swap:
        return -1
    return 0


def dcg_at_k(scores: np.ndarray, k: int = 10) -> float:
    """Discounted Cumulative Gain at k."""
    ranked_indices = np.argsort(-scores)[:k]
    gains = scores[ranked_indices]
    discounts = np.log2(np.arange(2, k + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(scores: np.ndarray, ideal_scores: np.ndarray, k: int = 10) -> float:
    """Normalised DCG at k."""
    dcg = dcg_at_k(scores, k)
    idcg = dcg_at_k(ideal_scores, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def mrr(scores: np.ndarray, target_idx: int) -> float:
    """Mean Reciprocal Rank of target document."""
    rank = int((scores > scores[target_idx]).sum()) + 1
    return 1.0 / rank


def exposure_at_position(rank: int, decay: str = "logarithmic") -> float:
    """Position-weighted exposure for a single document.

    Args:
        rank: 1-indexed rank position.
        decay: 'logarithmic' (1/log2(rank+1)) or 'linear' (1/rank).
    """
    if decay == "logarithmic":
        return 1.0 / np.log2(rank + 1)
    return 1.0 / rank


def compute_ranking_metrics(
    scores_original: np.ndarray,
    scores_swapped: np.ndarray,
    target_idx: int,
    k: int = 10,
) -> Dict[str, float]:
    """Compute all ranking impact metrics for a single counterfactual swap.

    Returns dict with: rank_flip, rank_displacement, topk_change,
                       mrr_orig, mrr_swap, mrr_change,
                       exposure_orig, exposure_swap.
    """
    flip = rank_flip_rate(scores_original, scores_swapped, target_idx)
    disp = rank_displacement(scores_original, scores_swapped, target_idx)
    topk = top_k_inclusion_change(scores_original, scores_swapped, target_idx, k)

    mrr_o = mrr(scores_original, target_idx)
    mrr_s = mrr(scores_swapped, target_idx)

    rank_o = int((scores_original > scores_original[target_idx]).sum()) + 1
    rank_s = int((scores_swapped > scores_swapped[target_idx]).sum()) + 1
    exp_o = exposure_at_position(rank_o)
    exp_s = exposure_at_position(rank_s)

    return {
        "rank_flip": flip,
        "rank_displacement": disp,
        "topk_change": topk,
        "mrr_original": mrr_o,
        "mrr_swapped": mrr_s,
        "mrr_change": mrr_s - mrr_o,
        "exposure_original": exp_o,
        "exposure_swapped": exp_s,
        "exposure_change": exp_s - exp_o,
    }
