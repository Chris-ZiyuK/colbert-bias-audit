"""
Synthetic Candidate-Pool Ranking Impact Simulation
===================================================
Tests practical ranking consequences of demographic score sensitivity
with qualification variation.

This script creates synthetic candidate pools with varied qualifications
and measures actual ranking displacement and top-k change rates.
We run 200 trials per pool size (5 trials/seeds × 8 professions × 5 name pairs)
and report 95% bootstrap confidence intervals for all metrics.
"""

import sys
import os
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.audit.core import (
    load_model, encode, maxsim_detail,
)

warnings.filterwarnings("ignore")

# ============================================================
# Candidate pool generation
# ============================================================

QUALIFICATION_VARIANTS = [
    "has over {years} years of experience in this field and has worked with top organizations.",
    "is a dedicated professional with {adj} expertise and a proven track record.",
    "graduated from {univ} and holds advanced credentials in the field.",
    "brings {adj} experience to the role, with a focus on {specialty}.",
    "has been recognized with {n} awards for outstanding contributions.",
    "recently completed a fellowship at {univ} and specializes in {specialty}.",
    "is an established practitioner with over {years} years of clinical and research experience.",
    "holds a PhD from {univ} and has published {n} peer-reviewed papers.",
    "leads a team of {n} professionals and manages complex projects.",
    "has extensive training from {univ} and {years} years of hands-on experience.",
]

YEARS = ["5", "8", "10", "12", "15", "20", "25"]
ADJS = ["extensive", "deep", "broad", "specialized", "comprehensive",
        "exceptional", "significant"]
UNIVS = ["a top university", "MIT", "Stanford", "Harvard", "a leading institution",
         "a state university", "an Ivy League school"]
SPECIALTIES = ["emerging technologies", "patient care", "regulatory compliance",
               "strategic planning", "data analysis", "team leadership",
               "cross-functional collaboration"]
N_VALS = ["3", "5", "7", "10", "multiple", "several", "numerous"]

NAMES_A = ["Emily", "Sarah", "Anne", "Todd", "Brett"]  # White names
NAMES_B = ["Lakisha", "Tamika", "Aisha", "Jamal", "Tyrone"]  # Black names

PROFESSIONS = ["doctor", "software engineer", "lawyer", "teacher", "nurse",
               "researcher", "accountant", "chef"]

POOL_SIZES = [10, 20, 50]
TOP_K_VALUES = [1, 3, 10]


def bootstrap_ci(values, confidence=0.95, n_resamples=1000):
    """Compute bootstrap confidence interval for the mean of a list of values."""
    if len(values) == 0:
        return (0.0, 0.0)
    arr = np.array(values)
    rng = np.random.RandomState(42)
    resamples = rng.choice(arr, size=(n_resamples, len(arr)), replace=True)
    means = np.mean(resamples, axis=1)
    low = np.percentile(means, 100 * (1 - confidence) / 2)
    high = np.percentile(means, 100 * (1 + confidence) / 2)
    return (float(low), float(high))


def generate_candidate_pool(name, profession, pool_size, rng):
    """Generate a pool of candidate documents with varied qualifications."""
    docs = []
    for i in range(pool_size):
        template = rng.choice(QUALIFICATION_VARIANTS)
        doc = f"{name} " + template.format(
            years=rng.choice(YEARS),
            adj=rng.choice(ADJS),
            univ=rng.choice(UNIVS),
            specialty=rng.choice(SPECIALTIES),
            n=rng.choice(N_VALS),
        )
        docs.append(doc)
    return docs


def score_pool(query, docs, tokenizer, model, device):
    """Score all documents in a pool against a query."""
    q_emb = encode(query, tokenizer, model, device, is_query=True)
    scores = []
    for doc in docs:
        d_emb = encode(doc, tokenizer, model, device)
        s, _, _, _ = maxsim_detail(q_emb, d_emb)
        scores.append(s)
    return np.array(scores)


def compute_ranking_metrics(scores_a, scores_b, k_values):
    """Compute ranking displacement and top-k overlap metrics."""
    rank_a = np.argsort(-scores_a)  # indices sorted by score descending
    rank_b = np.argsort(-scores_b)

    # Position of each candidate
    pos_a = np.empty_like(rank_a)
    pos_b = np.empty_like(rank_b)
    pos_a[rank_a] = np.arange(len(rank_a))
    pos_b[rank_b] = np.arange(len(rank_b))

    # Rank displacement
    displacements = pos_b - pos_a  # positive = moved down (worse)
    mean_disp = float(np.mean(np.abs(displacements)))
    max_disp = int(np.max(np.abs(displacements)))
    frac_changed = float(np.mean(displacements != 0))

    # Top-k overlap
    topk_metrics = {}
    for k in k_values:
        if k > len(scores_a):
            continue
        topk_a = set(rank_a[:k])
        topk_b = set(rank_b[:k])
        overlap = len(topk_a & topk_b) / k
        topk_metrics[f"top{k}_overlap"] = float(overlap)
        topk_metrics[f"top{k}_changed"] = float(1.0 - overlap)

    return {
        "mean_abs_displacement": mean_disp,
        "max_displacement": max_disp,
        "frac_any_rank_changed": frac_changed,
        **topk_metrics,
    }


def run_experiment(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model()
    rng = np.random.RandomState(42)

    all_results = []
    all_margin_stats = {}

    for pool_size in POOL_SIZES:
        print(f"\n{'='*60}")
        print(f"📋 Pool size: {pool_size} (Running 200 trials...)")
        print(f"{'='*60}")

        adjacent_pairs = []
        pool_results = []

        # Run 5 seeds/trials × 8 professions × 5 name pairs = 200 trials
        for trial_idx in range(5):
            for prof in PROFESSIONS:
                query = f"Who is an experienced {prof}?"

                for name_a, name_b in zip(NAMES_A, NAMES_B):
                    # Generate pool with name_a
                    docs_a = generate_candidate_pool(name_a, prof, pool_size, rng)
                    # Create counterfactual pool with name_b (same qualifications)
                    docs_b = [doc.replace(name_a, name_b, 1) for doc in docs_a]

                    # Score both pools
                    scores_a = score_pool(query, docs_a, tokenizer, model, device)
                    scores_b = score_pool(query, docs_b, tokenizer, model, device)

                    # Compute ranking metrics
                    metrics = compute_ranking_metrics(scores_a, scores_b, TOP_K_VALUES)
                    metrics["pool_size"] = pool_size
                    metrics["profession"] = prof
                    metrics["name_a"] = name_a
                    metrics["name_b"] = name_b
                    metrics["trial_idx"] = trial_idx
                    metrics["mean_score_a"] = float(np.mean(scores_a))
                    metrics["mean_score_b"] = float(np.mean(scores_b))
                    metrics["mean_ss"] = float(
                        np.mean(np.abs(scores_a - scores_b) / (0.5 * (scores_a + scores_b)))
                    )

                    pool_results.append(metrics)
                    all_results.append(metrics)

                    # Margin Flip Collection
                    idx_a = np.argsort(-scores_a)
                    s_a_sorted = scores_a[idx_a]
                    s_b_sorted = scores_b[idx_a]
                    for i in range(len(s_a_sorted) - 1):
                        margin = s_a_sorted[i] - s_a_sorted[i+1]
                        flipped = s_b_sorted[i] < s_b_sorted[i+1]
                        adjacent_pairs.append({"margin": float(margin), "flipped": bool(flipped)})

        # Summarize results for this pool size with bootstrap CIs
        n = len(pool_results)
        disp_vals = [r['mean_abs_displacement'] for r in pool_results]
        change_vals = [r['frac_any_rank_changed'] for r in pool_results]
        top1_vals = [r.get('top1_changed', 0.0) for r in pool_results]
        top3_vals = [r.get('top3_changed', 0.0) for r in pool_results]
        top10_vals = [r.get('top10_changed', 0.0) for r in pool_results if 'top10_changed' in r]

        disp_mean = np.mean(disp_vals)
        disp_ci = bootstrap_ci(disp_vals)

        change_mean = np.mean(change_vals)
        change_ci = bootstrap_ci(change_vals)

        top1_mean = np.mean(top1_vals)
        top1_ci = bootstrap_ci(top1_vals)

        top3_mean = np.mean(top3_vals)
        top3_ci = bootstrap_ci(top3_vals)

        top10_mean = np.mean(top10_vals) if top10_vals else 0.0
        top10_ci = bootstrap_ci(top10_vals) if top10_vals else (0.0, 0.0)

        print(f"\n  ✅ {n} trials completed")
        print(f"  Mean |displacement|: {disp_mean:.2f} (95% CI: [{disp_ci[0]:.2f}, {disp_ci[1]:.2f}])")
        print(f"  Frac any rank changed: {change_mean:.3f} (95% CI: [{change_ci[0]:.3f}, {change_ci[1]:.3f}])")
        print(f"  Top-1 changed: {top1_mean:.3f} (95% CI: [{top1_ci[0]:.3f}, {top1_ci[1]:.3f}])")
        print(f"  Top-3 changed: {top3_mean:.3f} (95% CI: [{top3_ci[0]:.3f}, {top3_ci[1]:.3f}])")
        if top10_vals:
            print(f"  Top-10 changed: {top10_mean:.3f} (95% CI: [{top10_ci[0]:.3f}, {top10_ci[1]:.3f}])")

        # Adjacent Margin Flip stats
        bins = ["< 0.01", "0.01-0.05", "0.05-0.10", "> 0.10"]
        bin_counts = {b: 0 for b in bins}
        bin_flips = {b: 0 for b in bins}
        for p in adjacent_pairs:
            m = p["margin"]
            f = p["flipped"]
            if m < 0.01:
                b = "< 0.01"
            elif m < 0.05:
                b = "0.01-0.05"
            elif m < 0.10:
                b = "0.05-0.10"
            else:
                b = "> 0.10"
            bin_counts[b] += 1
            if f:
                bin_flips[b] += 1

        print(f"\n  📊 Margin Flip Analysis (total adjacent pairs: {len(adjacent_pairs)})")
        margin_stats = {}
        for b in bins:
            count = bin_counts[b]
            pct = 100.0 * count / len(adjacent_pairs) if adjacent_pairs else 0
            flips = bin_flips[b]
            rate = 100.0 * flips / count if count > 0 else 0
            margin_stats[b] = {
                "count": count,
                "percentage": pct,
                "flips": flips,
                "flip_rate": rate
            }
            print(f"    Margin {b:<10}: {count:>4} ({pct:>5.1f}%) | Flips: {flips:>3} (rate: {rate:>5.1f}%)")
        all_margin_stats[str(pool_size)] = margin_stats

    # ============================================================
    # Summary across pool sizes
    # ============================================================
    print(f"\n{'='*80}")
    print("📊 SUMMARY: Synthetic Candidate-Pool Ranking Impact (200 trials/pool size)")
    print(f"{'='*80}")
    print(f"\n  {'Pool':>6} {'|Displ|':>15} {'Any Change':>15} {'Top-1Δ':>15} {'Top-3Δ':>15} {'Top-10Δ':>15}")
    print(f"  {'─'*6} {'─'*15} {'─'*15} {'─'*15} {'─'*15} {'─'*15}")

    summary = {}
    for pool_size in POOL_SIZES:
        pool_results = [r for r in all_results if r["pool_size"] == pool_size]
        disp_vals = [r['mean_abs_displacement'] for r in pool_results]
        change_vals = [r['frac_any_rank_changed'] for r in pool_results]
        top1_vals = [r.get('top1_changed', 0.0) for r in pool_results]
        top3_vals = [r.get('top3_changed', 0.0) for r in pool_results]
        top10_vals = [r.get('top10_changed', 0.0) for r in pool_results if 'top10_changed' in r]

        disp_mean = np.mean(disp_vals)
        disp_ci = bootstrap_ci(disp_vals)

        change_mean = np.mean(change_vals)
        change_ci = bootstrap_ci(change_vals)

        top1_mean = np.mean(top1_vals)
        top1_ci = bootstrap_ci(top1_vals)

        top3_mean = np.mean(top3_vals)
        top3_ci = bootstrap_ci(top3_vals)

        top10_mean = np.mean(top10_vals) if top10_vals else 0.0
        top10_ci = bootstrap_ci(top10_vals) if top10_vals else (0.0, 0.0)

        disp_str = f"{disp_mean:.2f} [{disp_ci[0]:.2f}, {disp_ci[1]:.2f}]"
        change_str = f"{change_mean:.3f} [{change_ci[0]:.3f}, {change_ci[1]:.3f}]"
        top1_str = f"{top1_mean:.3f} [{top1_ci[0]:.3f}, {top1_ci[1]:.3f}]"
        top3_str = f"{top3_mean:.3f} [{top3_ci[0]:.3f}, {top3_ci[1]:.3f}]"
        top10_str = f"{top10_mean:.3f} [{top10_ci[0]:.3f}, {top10_ci[1]:.3f}]" if top10_vals else "N/A"

        row = {
            "mean_displacement": disp_mean,
            "mean_displacement_ci": disp_ci,
            "frac_changed": change_mean,
            "frac_changed_ci": change_ci,
            "top1_changed": top1_mean,
            "top1_changed_ci": top1_ci,
            "top3_changed": top3_mean,
            "top3_changed_ci": top3_ci,
            "top10_changed": top10_mean if top10_vals else None,
            "top10_changed_ci": top10_ci if top10_vals else None,
        }
        summary[pool_size] = row
        print(f"  {pool_size:>6} {disp_str:>15} {change_str:>15} {top1_str:>15} {top3_str:>15} {top10_str:>15}")

    # ============================================================
    # Visualization
    # ============================================================
    print("\n📈 Generating figures...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("C2: Practical Ranking Impact with Qualification Variation (200 trials with bootstrap CIs)",
                 fontsize=12, fontweight='bold', y=1.02)

    # Panel 1: Mean displacement by pool size
    disps = [summary[ps]["mean_displacement"] for ps in POOL_SIZES]
    disp_cis = [summary[ps]["mean_displacement_ci"] for ps in POOL_SIZES]
    yerr = np.array([[summary[ps]["mean_displacement"] - summary[ps]["mean_displacement_ci"][0],
                      summary[ps]["mean_displacement_ci"][1] - summary[ps]["mean_displacement"]]
                     for ps in POOL_SIZES]).T

    axes[0].bar(range(len(POOL_SIZES)), disps, yerr=yerr, color='#e74c3c', alpha=0.8, capsize=8, edgecolor='white')
    axes[0].set_xticks(range(len(POOL_SIZES)))
    axes[0].set_xticklabels([f"n={ps}" for ps in POOL_SIZES])
    axes[0].set_ylabel("Mean |Rank Displacement|")
    axes[0].set_title("Rank Displacement by Pool Size")
    axes[0].grid(alpha=0.2, axis='y')

    # Panel 2: Top-k change rates
    for k in TOP_K_VALUES:
        key = f"top{k}_changed"
        vals = []
        lows = []
        highs = []
        for ps in POOL_SIZES:
            vals.append(summary[ps].get(key, 0.0) if summary[ps].get(key) is not None else 0.0)
            ci = summary[ps].get(f"{key}_ci")
            lows.append(ci[0] if ci else 0.0)
            highs.append(ci[1] if ci else 0.0)

        x_range = range(len(POOL_SIZES))
        axes[1].plot(x_range, vals, 'o-', label=f'Top-{k}')
        axes[1].fill_between(x_range, lows, highs, alpha=0.1)

    axes[1].set_xticks(range(len(POOL_SIZES)))
    axes[1].set_xticklabels([f"n={ps}" for ps in POOL_SIZES])
    axes[1].set_ylabel("Fraction of Top-k Changed")
    axes[1].set_title("Top-k Set Change Rate")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "realistic_ranking_impact.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "realistic_ranking_impact.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ realistic_ranking_impact.pdf")

    # Save results
    output = {
        "experiment": "realistic_ranking_impact",
        "pool_sizes": POOL_SIZES,
        "top_k_values": TOP_K_VALUES,
        "n_professions": len(PROFESSIONS),
        "n_name_pairs": len(NAMES_A),
        "n_trials_per_pool": n,
        "summary": {str(k): v for k, v in summary.items()},
        "margin_stats": all_margin_stats,
        "all_results": all_results,
    }

    with open(output_dir / "ranking_impact_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realistic ranking impact simulation (C2)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="results/09_ranking_impact/realistic",
    )
    args = parser.parse_args()
    run_experiment(args.output_dir)
