"""
Statistical tests used throughout the bias audit experiments.

All tests follow the conventions in the paper:
  - Mann-Whitney U for independent samples (function vs content TCD)
  - Wilcoxon signed-rank for paired samples (within-pair Func > Cont)
  - OLS regression for confound decomposition (with clustered SEs)
  - Cohen's d for effect size
  - ANOVA for multi-group comparisons (rarity categories)
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats


def mann_whitney_one_sided(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    alternative: str = "greater",
) -> Dict[str, float]:
    """One-sided Mann-Whitney U test.

    Args:
        sample_a: First sample (hypothesised to be larger if alternative='greater').
        sample_b: Second sample.
        alternative: 'greater', 'less', or 'two-sided'.

    Returns:
        Dict with statistic, p_value, effect_size (rank-biserial correlation).
    """
    u_stat, p_val = stats.mannwhitneyu(
        sample_a, sample_b, alternative=alternative
    )
    n1, n2 = len(sample_a), len(sample_b)
    # Rank-biserial correlation as effect size
    rbc = 1 - (2 * u_stat) / (n1 * n2)

    return {
        "statistic": float(u_stat),
        "p_value": float(p_val),
        "effect_size_rbc": float(rbc),
        "n_a": n1,
        "n_b": n2,
    }


def wilcoxon_signed_rank(
    paired_a: np.ndarray,
    paired_b: np.ndarray,
    alternative: str = "greater",
) -> Dict[str, float]:
    """Wilcoxon signed-rank test for paired samples.

    Args:
        paired_a: First paired sample (e.g., Func-TCD per pair).
        paired_b: Second paired sample (e.g., Cont-TCD per pair).
        alternative: 'greater', 'less', or 'two-sided'.

    Returns:
        Dict with statistic, p_value, n_pairs.
    """
    stat, p_val = stats.wilcoxon(
        paired_a, paired_b, alternative=alternative
    )
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "n_pairs": len(paired_a),
    }


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size for two independent samples."""
    n_a, n_b = len(group_a), len(group_b)
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def one_way_anova(*groups: np.ndarray) -> Dict[str, float]:
    """One-way ANOVA across multiple groups."""
    f_stat, p_val = stats.f_oneway(*groups)
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "n_groups": len(groups),
    }


def binomial_direction_test(
    tcd_array: np.ndarray,
    direction: str = "positive",
) -> Dict[str, float]:
    """Binomial test for directional bias.

    Tests whether TCD values are systematically positive (favouring doc A)
    or negative (favouring doc B) more than chance (50%).

    Args:
        tcd_array: Signed TCD values.
        direction: 'positive' or 'negative'.

    Returns:
        Dict with n_positive, n_negative, fraction, p_value.
    """
    n_pos = int(np.sum(tcd_array > 0))
    n_neg = int(np.sum(tcd_array < 0))
    n_total = n_pos + n_neg
    if n_total == 0:
        return {"n_positive": 0, "n_negative": 0, "fraction": 0.0, "p_value": 1.0}

    if direction == "positive":
        k = n_pos
    else:
        k = n_neg

    p_val = stats.binom_test(k, n_total, 0.5, alternative="greater")

    return {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "fraction": k / n_total,
        "p_value": float(p_val),
        "n_total": n_total,
    }


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    point = float(statistic_fn(data))
    boot_stats = np.array([
        statistic_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return point, lo, hi
