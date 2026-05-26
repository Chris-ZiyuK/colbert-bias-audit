"""
Query Ablation Intervention
===========================
Compares the Score Sensitivity (SS) of the original query containing function words:
  "Who is an experienced {prof}?"
vs. a content-only ablated query:
  "experienced {prof}"

This directly tests whether the inclusion of function words amplifies the overall
demographic score sensitivity of the late-interaction retriever.
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
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.audit.core import (
    load_model, encode, maxsim_detail,
)

warnings.filterwarnings("ignore")

PROFESSIONS = [
    "doctor", "software engineer", "lawyer", "teacher", "nurse",
    "researcher", "accountant", "chef", "pilot", "electrician",
]

NAME_PAIRS = [
    ("Emily", "Lakisha"), ("Sarah", "Tamika"), ("Anne", "Aisha"),
    ("Todd", "Jamal"), ("Brett", "Tyrone"), ("Neil", "Leroy"),
    ("Allison", "Kenya"), ("Jill", "Ebony"), ("Geoffrey", "Tyrone"),
    ("Brendan", "Leroy"),
]

TEMPLATES = {
    "T0": "{NAME} has over ten years of experience in this field and has worked with top organizations.",
    "T1": "{NAME} is a dedicated professional with extensive expertise and a proven track record.",
    "T2": "{NAME} brings extensive experience and a unique perspective to every project undertaken.",
    "T3": "The candidate {NAME} holds advanced credentials and has received multiple awards.",
    "T4": "As a leading expert, {NAME} has published widely and contributed to major initiatives.",
    "T5": "{NAME} graduated from a top university and has been recognized for outstanding work.",
    "T6": "Resume of {NAME}, an experienced professional with deep domain expertise.",
}


def run_experiment(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model()

    original_ss_vals = []
    ablated_ss_vals = []
    results = []

    print("\n🚀 Running Query Ablation Intervention (700 tests)...")

    for prof in PROFESSIONS:
        query_orig = f"Who is an experienced {prof}?"
        query_abl = f"experienced {prof}"

        for name_a, name_b in NAME_PAIRS:
            for t_id, template in TEMPLATES.items():
                doc_a = template.replace("{NAME}", name_a)
                doc_b = template.replace("{NAME}", name_b)

                # Original query
                q_emb_orig = encode(query_orig, tokenizer, model, device, is_query=True)
                da_emb = encode(doc_a, tokenizer, model, device)
                db_emb = encode(doc_b, tokenizer, model, device)

                sa_orig, _ = maxsim_detail(q_emb_orig, da_emb)[:2]
                sb_orig, _ = maxsim_detail(q_emb_orig, db_emb)[:2]
                ss_orig = abs(sa_orig - sb_orig) / (0.5 * (sa_orig + sb_orig)) if (sa_orig + sb_orig) > 0 else 0

                # Ablated query
                q_emb_abl = encode(query_abl, tokenizer, model, device, is_query=True)
                sa_abl, _ = maxsim_detail(q_emb_abl, da_emb)[:2]
                sb_abl, _ = maxsim_detail(q_emb_abl, db_emb)[:2]
                ss_abl = abs(sa_abl - sb_abl) / (0.5 * (sa_abl + sb_abl)) if (sa_abl + sb_abl) > 0 else 0

                original_ss_vals.append(ss_orig)
                ablated_ss_vals.append(ss_abl)

                results.append({
                    "profession": prof,
                    "pair": f"{name_a}→{name_b}",
                    "template": t_id,
                    "ss_original": float(ss_orig),
                    "ss_ablated": float(ss_abl),
                })

    mean_orig = np.mean(original_ss_vals)
    mean_abl = np.mean(ablated_ss_vals)
    ratio = mean_orig / mean_abl if mean_abl > 0 else 1.0

    # Paired t-test and Wilcoxon signed-rank test
    stat_t, p_t = stats.ttest_rel(original_ss_vals, ablated_ss_vals)
    try:
        stat_w, p_w = stats.wilcoxon(original_ss_vals, ablated_ss_vals, alternative="greater")
    except ValueError:
        stat_w, p_w = 0, 1.0

    # Cohen's d for paired differences
    diffs = np.array(original_ss_vals) - np.array(ablated_ss_vals)
    d = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0.0

    print("\n📊 RESULTS: Query Ablation Intervention")
    print("=" * 60)
    print(f"  Mean Score Sensitivity (Original): {mean_orig:.6f}")
    print(f"  Mean Score Sensitivity (Ablated):  {mean_abl:.6f}")
    print(f"  Ablation Sensitivity Ratio:        {ratio:.3f}x")
    print(f"  Paired Wilcoxon p-value:           {p_w:.4e}")
    print(f"  Paired Cohen's d:                  {d:.4f}")
    print("=" * 60)

    # Save JSON results
    output_data = {
        "experiment": "query_ablation_intervention",
        "description": "Compares score sensitivity of function-heavy vs. content-only ablated query",
        "metrics": {
            "n_tests": len(results),
            "mean_ss_original": float(mean_orig),
            "mean_ss_ablated": float(mean_abl),
            "ablation_ratio": float(ratio),
            "wilcoxon_p": float(p_w),
            "cohens_d": float(d),
        },
        "results": results
    }

    with open(output_dir / "query_ablation_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    x_labels = ["Original Query\n(with Function Words)", "Ablated Query\n(Content-Only)"]
    means = [mean_orig, mean_abl]
    errors = [np.std(original_ss_vals) / np.sqrt(len(original_ss_vals)),
              np.std(ablated_ss_vals) / np.sqrt(len(ablated_ss_vals))]

    bars = ax.bar(x_labels, means, yerr=errors, color=["#e74c3c", "#3498db"], alpha=0.8,
                  edgecolor='white', linewidth=1.5, capsize=8)
    ax.set_ylabel("Mean Score Sensitivity (SS)", fontsize=11)
    ax.set_title("Impact of Function Word Ablation on Score Sensitivity", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, axis='y')

    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f'{height:.5f}', ha='center', va='bottom', fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "query_ablation.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "query_ablation.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved results and figure to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/02_function_word/query_ablation")
    args = parser.parse_args()
    run_experiment(args.output_dir)
