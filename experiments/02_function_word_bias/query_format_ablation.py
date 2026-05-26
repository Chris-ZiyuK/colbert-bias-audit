"""
Query Format Ablation — Addressing Reviewer D1
================================================
Tests whether the function-word bias pattern holds across different query formats,
not just "Who is an experienced {profession}?"

Design: 5 query formats with varying function/content word compositions.

Usage:
  python experiments/02_function_word_bias/query_format_ablation.py \
    --output-dir results/02_function_word/query_format_ablation
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
    load_model, encode, get_tokens, maxsim_detail, classify_token,
)
from src.metrics.stats import cohens_d, wilcoxon_signed_rank

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

QUERY_FORMATS = {
    "Q1_original": {
        "template": "Who is an experienced {prof}?",
        "description": "Original (4 function words: who, is, an, ?)",
    },
    "Q2_with_clause": {
        "template": "{prof} with extensive professional experience",
        "description": "Prepositional (1 function word: with)",
    },
    "Q3_imperative": {
        "template": "Find a qualified {prof} for hire",
        "description": "Imperative (2 function words: a, for)",
    },
    "Q4_keyword": {
        "template": "experienced {prof} resume qualifications",
        "description": "Keyword-only (0 function words)",
    },
    "Q5_what_makes": {
        "template": "What makes a good {prof}?",
        "description": "Question variant (3 function words: what, a, ?)",
    },
}

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


def compute_tcd_for_pair(query, doc_a, doc_b, tokenizer, model, device):
    """Compute Func-TCD, Cont-TCD for a pair."""
    q_emb = encode(query, tokenizer, model, device, is_query=True)
    q_tokens = get_tokens(query, tokenizer, is_query=True)
    da_emb = encode(doc_a, tokenizer, model, device)
    db_emb = encode(doc_b, tokenizer, model, device)

    sa, _, scores_a, _ = maxsim_detail(q_emb, da_emb)
    sb, _, scores_b, _ = maxsim_detail(q_emb, db_emb)

    tcd = scores_a - scores_b
    ss = abs(sa - sb) / (0.5 * (sa + sb)) if (sa + sb) > 0 else 0

    func_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens)
                 if classify_token(t) == "function"]
    cont_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens)
                 if classify_token(t) == "content"]

    return {
        "ss": ss,
        "func_tcd": float(np.mean(func_tcds)) if func_tcds else 0.0,
        "cont_tcd": float(np.mean(cont_tcds)) if cont_tcds else 0.0,
        "n_func": len(func_tcds),
        "n_cont": len(cont_tcds),
        "ratio": (float(np.mean(func_tcds)) / float(np.mean(cont_tcds))
                  if cont_tcds and np.mean(cont_tcds) > 0 else float('inf'))
                 if func_tcds else 0.0,
        "q_tokens": q_tokens,
    }


def run_experiment(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model()

    all_results = {}
    summaries = {}

    for qf_name, qf in QUERY_FORMATS.items():
        print(f"\n{'='*60}")
        print(f"📋 Query format: {qf_name} — {qf['description']}")
        print(f"   Template: \"{qf['template']}\"")
        print(f"{'='*60}")

        results = []
        for prof in PROFESSIONS:
            query = qf["template"].format(prof=prof)
            for name_a, name_b in NAME_PAIRS:
                for t_id, template in TEMPLATES.items():
                    doc_a = template.replace("{NAME}", name_a)
                    doc_b = template.replace("{NAME}", name_b)

                    r = compute_tcd_for_pair(
                        query, doc_a, doc_b, tokenizer, model, device
                    )
                    r["profession"] = prof
                    r["pair"] = f"{name_a}→{name_b}"
                    r["template"] = t_id
                    r["query_format"] = qf_name
                    # Remove non-serializable token list for storage
                    r.pop("q_tokens", None)
                    results.append(r)

        n = len(results)
        func_means = [r["func_tcd"] for r in results]
        cont_means = [r["cont_tcd"] for r in results]
        mean_func = np.mean(func_means)
        mean_cont = np.mean(cont_means)
        mean_ratio = mean_func / mean_cont if mean_cont > 0 else float('inf')
        mean_n_func = np.mean([r["n_func"] for r in results])
        mean_n_cont = np.mean([r["n_cont"] for r in results])

        # Statistical test
        if len(func_means) > 10 and mean_n_func > 0:
            try:
                w = wilcoxon_signed_rank(
                    np.array(func_means), np.array(cont_means), alternative="greater"
                )
                w_p = w["p_value"]
            except Exception:
                w_p = 1.0
            d = cohens_d(np.array(func_means), np.array(cont_means))
        else:
            w_p = 1.0
            d = 0.0

        summaries[qf_name] = {
            "n_tests": n,
            "mean_func_tcd": float(mean_func),
            "mean_cont_tcd": float(mean_cont),
            "func_cont_ratio": float(mean_ratio) if mean_ratio != float('inf') else None,
            "mean_n_func_tokens": float(mean_n_func),
            "mean_n_cont_tokens": float(mean_n_cont),
            "wilcoxon_p": float(w_p),
            "cohens_d": float(d),
            "description": qf["description"],
        }

        all_results[qf_name] = results

        print(f"  ✅ {n} tests")
        print(f"     Func tokens/query: {mean_n_func:.1f}, Cont tokens/query: {mean_n_cont:.1f}")
        print(f"     Mean Func-TCD = {mean_func:.6f}")
        print(f"     Mean Cont-TCD = {mean_cont:.6f}")
        print(f"     F/C Ratio     = {mean_ratio:.3f}×" if mean_ratio != float('inf')
              else "     F/C Ratio     = N/A (no function words)")
        print(f"     Wilcoxon p    = {w_p:.4e}")
        print(f"     Cohen's d     = {d:.4f}")

    # ============================================================
    # Summary table
    # ============================================================
    print(f"\n{'='*70}")
    print("📊 SUMMARY: Function-Word Pattern Across Query Formats")
    print(f"{'='*70}")
    print(f"\n  {'Format':<18} {'#Func':>6} {'#Cont':>6} {'F/C Ratio':>10} "
          f"{'d':>8} {'p':>12} {'Pattern?':>10}")
    print(f"  {'─'*18} {'─'*6} {'─'*6} {'─'*10} {'─'*8} {'─'*12} {'─'*10}")

    for qf_name in QUERY_FORMATS:
        s = summaries[qf_name]
        ratio_str = f"{s['func_cont_ratio']:.3f}×" if s['func_cont_ratio'] else "N/A"
        pattern = "✅ Yes" if (s['func_cont_ratio'] and s['func_cont_ratio'] > 1.0
                               and s['wilcoxon_p'] < 0.05) else "❌ No"
        if s['mean_n_func_tokens'] < 0.5:
            pattern = "— (no func)"
        print(f"  {qf_name:<18} {s['mean_n_func_tokens']:>6.1f} {s['mean_n_cont_tokens']:>6.1f} "
              f"{ratio_str:>10} {s['cohens_d']:>8.4f} {s['wilcoxon_p']:>12.4e} {pattern:>10}")

    # ============================================================
    # Visualization
    # ============================================================
    print("\n📈 Generating figures...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("D1: Function-Word Pattern Across Query Formats",
                 fontsize=13, fontweight='bold', y=1.02)

    qf_names = list(QUERY_FORMATS.keys())
    labels = [QUERY_FORMATS[q]["description"].split(" (")[0] for q in qf_names]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # Panel 1: F/C Ratio
    ratios = []
    for q in qf_names:
        r = summaries[q]["func_cont_ratio"]
        ratios.append(r if r else 0)

    bars = axes[0].bar(range(len(qf_names)), ratios, color=colors, alpha=0.8,
                       edgecolor='white', linewidth=1.5)
    axes[0].axhline(y=1.0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    axes[0].set_xticks(range(len(qf_names)))
    axes[0].set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    axes[0].set_ylabel("Func / Cont TCD Ratio", fontsize=11)
    axes[0].set_title("Func/Cont Ratio by Query Format", fontsize=11)
    axes[0].grid(alpha=0.2, axis='y')

    for i, (r, bar) in enumerate(zip(ratios, bars)):
        if r > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                         f'{r:.2f}×', ha='center', va='bottom', fontsize=9,
                         fontweight='bold')

    # Panel 2: Grouped bar (Func vs Cont TCD)
    x = np.arange(len(qf_names))
    width = 0.35
    func_vals = [summaries[q]["mean_func_tcd"] for q in qf_names]
    cont_vals = [summaries[q]["mean_cont_tcd"] for q in qf_names]
    axes[1].bar(x - width/2, func_vals, width, label='Function Words',
                color='#e74c3c', alpha=0.7)
    axes[1].bar(x + width/2, cont_vals, width, label='Content Words',
                color='#3498db', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    axes[1].set_ylabel("Mean |TCD|", fontsize=11)
    axes[1].set_title("Func vs Cont TCD by Query Format", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "query_format_ablation.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "query_format_ablation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ query_format_ablation.pdf")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        "experiment": "query_format_ablation",
        "description": "Tests function-word pattern across 5 query formats (D1)",
        "summaries": summaries,
    }

    with open(output_dir / "query_format_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query format ablation experiment (D1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/02_function_word/query_format_ablation",
    )
    args = parser.parse_args()
    run_experiment(args.output_dir)
