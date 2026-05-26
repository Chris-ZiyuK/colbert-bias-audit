"""
Non-Identity Perturbation Experiment — Addressing Reviewer C1
==============================================================
Core question: Is function-word TCD amplification SPECIFIC to identity swaps,
or a general property of MaxSim sensitivity?

Design: 5 perturbation types on the same queries/templates:
  1. IDENTITY:    Emily → Lakisha       (cross-race name swap)
  2. NUMBER:      ten → fifteen         (numeric word swap)
  3. ADJECTIVE:   dedicated → committed (semantic adjective swap)
  4. INSTITUTION: Harvard → Stanford    (named entity swap)
  5. RARE_NONID:  Emily → Zephyx        (rare nonsense word, not a name)

If identity swaps show a HIGHER Func/Cont ratio than other perturbation
types, the function-word effect is identity-specific. If all types show
similar ratios, it's a general MaxSim property.

Usage:
  python experiments/01_control_validation/non_identity_perturbation.py \
    --output-dir results/01_control/non_identity_perturbation
"""

import sys
import os
import argparse
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy import stats

# Project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.audit.core import (
    load_model, encode, get_tokens, maxsim_detail, classify_token
)
from src.metrics.stats import cohens_d, wilcoxon_signed_rank, mann_whitney_one_sided

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

QUERIES = [
    "Who is an experienced doctor?",
    "Who is an experienced software engineer?",
    "Who is an experienced lawyer?",
    "Who is an experienced teacher?",
    "Who is an experienced nurse?",
    "Who is an experienced researcher?",
    "Who is an experienced accountant?",
    "Who is an experienced chef?",
]

# Paper's 7 templates (Appendix B)
TEMPLATES = {
    "T0": "{X} has over ten years of experience in this field and has worked with top organizations.",
    "T1": "{X} is a dedicated professional with extensive expertise and a proven track record.",
    "T2": "{X} brings extensive experience and a unique perspective to every project undertaken.",
    "T3": "The candidate {X} holds advanced credentials and has received multiple awards.",
    "T4": "As a leading expert, {X} has published widely and contributed to major initiatives.",
    "T5": "{X} graduated from a top university and has been recognized for outstanding work.",
    "T6": "Resume of {X}, an experienced professional with deep domain expertise.",
}

# For institution swaps, we need templates containing institution names
INST_TEMPLATES = {
    "T5_inst": "{X} graduated from {INST} and has been recognized for outstanding work.",
    "T0_inst": "{X} has over ten years of experience at {INST} and has worked with top organizations.",
    "T1_inst": "{X} is a dedicated professional at {INST} with extensive expertise and a proven track record.",
}

# ============================================================
# 5 Perturbation conditions
# ============================================================

PERTURBATIONS = {
    "IDENTITY": {
        "description": "Cross-race identity name swap",
        "pairs": [
            ("Emily", "Lakisha"),
            ("Sarah", "Tamika"),
            ("Anne", "Aisha"),
            ("Todd", "Jamal"),
            ("Brett", "Tyrone"),
        ],
        "swap_field": "X",  # Replace {X} in template
        "use_templates": "standard",
    },
    "NUMBER": {
        "description": "Numeric word swap in template T0",
        "pairs": [
            ("ten", "fifteen"),
            ("ten", "twenty"),
            ("ten", "five"),
            ("ten", "thirty"),
            ("ten", "twelve"),
        ],
        "swap_field": "inline",  # Replace within template text
        "use_templates": "standard",
    },
    "ADJECTIVE": {
        "description": "Adjective swap in templates",
        "pairs": [
            ("dedicated", "committed"),
            ("extensive", "broad"),
            ("leading", "prominent"),
            ("outstanding", "remarkable"),
            ("advanced", "superior"),
        ],
        "swap_field": "inline",
        "use_templates": "standard",
    },
    "INSTITUTION": {
        "description": "Institution named-entity swap",
        "pairs": [
            ("Harvard", "Stanford"),
            ("Google", "Microsoft"),
            ("MIT", "Princeton"),
            ("Yale", "Columbia"),
            ("Cambridge", "Oxford"),
        ],
        "swap_field": "INST",
        "use_templates": "institution",
    },
    "RARE_NONID": {
        "description": "Rare nonsense words (non-identity, high BPE fragmentation)",
        "pairs": [
            ("Emily", "Zephyrox"),
            ("Emily", "Quorveline"),
            ("Emily", "Brinthalux"),
            ("Emily", "Colvantrix"),
            ("Emily", "Tasperian"),
        ],
        "swap_field": "X",
        "use_templates": "standard",
    },
}

# ============================================================
# Core computation
# ============================================================

def compute_tcd_for_pair(query, doc_a, doc_b, tokenizer, model, device):
    """Compute Func-TCD, Cont-TCD, and SS for a document pair."""
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

    func_mean = float(np.mean(func_tcds)) if func_tcds else 0.0
    cont_mean = float(np.mean(cont_tcds)) if cont_tcds else 0.0

    return {
        "ss": ss,
        "func_tcd": func_mean,
        "cont_tcd": cont_mean,
        "ratio": func_mean / cont_mean if cont_mean > 0 else float('inf'),
        "func_tcds": func_tcds,
        "cont_tcds": cont_tcds,
    }


def generate_doc_pair(template_text, pair, swap_field, base_name="Emily"):
    """Generate a counterfactual document pair from a template and swap pair."""
    word_a, word_b = pair

    if swap_field == "X":
        doc_a = template_text.replace("{X}", word_a)
        doc_b = template_text.replace("{X}", word_b)
    elif swap_field == "INST":
        # Institution swap: use base_name for {X}, swap {INST}
        doc_a = template_text.replace("{X}", base_name).replace("{INST}", word_a)
        doc_b = template_text.replace("{X}", base_name).replace("{INST}", word_b)
    elif swap_field == "inline":
        # Inline text swap: replace within the template text directly
        if word_a in template_text:
            doc_a = template_text.replace("{X}", base_name)
            doc_b = template_text.replace("{X}", base_name).replace(word_a, word_b, 1)
        else:
            return None, None  # Word not in this template
    else:
        return None, None

    return doc_a, doc_b


# ============================================================
# Main experiment
# ============================================================

def run_experiment(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model()

    all_results = {}
    condition_summaries = {}

    for cond_name, cond in PERTURBATIONS.items():
        print(f"\n{'='*60}")
        print(f"📋 Condition: {cond_name} — {cond['description']}")
        print(f"{'='*60}")

        if cond["use_templates"] == "institution":
            templates = INST_TEMPLATES
        else:
            templates = TEMPLATES

        results = []
        all_func_tcds = []
        all_cont_tcds = []

        for pair in cond["pairs"]:
            for query in QUERIES:
                for t_id, template_text in templates.items():
                    doc_a, doc_b = generate_doc_pair(
                        template_text, pair, cond["swap_field"]
                    )
                    if doc_a is None:
                        continue

                    r = compute_tcd_for_pair(
                        query, doc_a, doc_b, tokenizer, model, device
                    )
                    r["pair"] = f"{pair[0]}→{pair[1]}"
                    r["query"] = query
                    r["template"] = t_id
                    results.append(r)
                    all_func_tcds.extend(r["func_tcds"])
                    all_cont_tcds.extend(r["cont_tcds"])

        # Compute summary statistics
        n = len(results)
        func_means = [r["func_tcd"] for r in results]
        cont_means = [r["cont_tcd"] for r in results]
        ss_vals = [r["ss"] for r in results]

        mean_func = np.mean(func_means) if func_means else 0
        mean_cont = np.mean(cont_means) if cont_means else 0
        mean_ratio = mean_func / mean_cont if mean_cont > 0 else float('inf')
        mean_ss = np.mean(ss_vals)

        # Paired Wilcoxon test: Func-TCD > Cont-TCD?
        if len(func_means) > 10:
            try:
                w_result = wilcoxon_signed_rank(
                    np.array(func_means), np.array(cont_means), alternative="greater"
                )
                w_p = w_result["p_value"]
            except Exception:
                w_p = 1.0
            d = cohens_d(np.array(func_means), np.array(cont_means))
        else:
            w_p = 1.0
            d = 0.0

        condition_summaries[cond_name] = {
            "n_tests": n,
            "mean_func_tcd": float(mean_func),
            "mean_cont_tcd": float(mean_cont),
            "func_cont_ratio": float(mean_ratio),
            "mean_ss": float(mean_ss),
            "wilcoxon_p": float(w_p),
            "cohens_d": float(d),
        }

        # Strip non-serializable lists for JSON
        all_results[cond_name] = [
            {k: v for k, v in r.items() if k not in ("func_tcds", "cont_tcds")}
            for r in results
        ]

        print(f"  ✅ {n} tests")
        print(f"     Mean Func-TCD = {mean_func:.6f}")
        print(f"     Mean Cont-TCD = {mean_cont:.6f}")
        print(f"     F/C Ratio     = {mean_ratio:.3f}×")
        print(f"     Mean SS       = {mean_ss:.6f}")
        print(f"     Wilcoxon p    = {w_p:.6e}")
        print(f"     Cohen's d     = {d:.4f}")

    # ============================================================
    # Cross-condition comparison
    # ============================================================
    print(f"\n{'='*70}")
    print("📊 CROSS-CONDITION COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Condition':<15} {'N':>6} {'F/C Ratio':>10} {'Func-TCD':>10} "
          f"{'Cont-TCD':>10} {'SS':>10} {'p':>12} {'d':>8}")
    print(f"  {'─'*15} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*8}")

    for cond_name in PERTURBATIONS:
        s = condition_summaries[cond_name]
        sig = "***" if s["wilcoxon_p"] < 0.001 else "**" if s["wilcoxon_p"] < 0.01 else "*" if s["wilcoxon_p"] < 0.05 else "n.s."
        print(f"  {cond_name:<15} {s['n_tests']:>6} {s['func_cont_ratio']:>10.3f}× "
              f"{s['mean_func_tcd']:>10.6f} {s['mean_cont_tcd']:>10.6f} "
              f"{s['mean_ss']:>10.6f} {s['wilcoxon_p']:>12.2e} {s['cohens_d']:>7.4f} {sig}")

    # ============================================================
    # Key diagnostic: Is identity ratio significantly higher?
    # ============================================================
    identity_ratios = [r["ratio"] for r in all_results["IDENTITY"]
                       if r["ratio"] != float('inf')]
    non_id_ratios = []
    for cond_name in ["NUMBER", "ADJECTIVE", "INSTITUTION", "RARE_NONID"]:
        non_id_ratios.extend([
            r["ratio"] for r in all_results[cond_name]
            if r["ratio"] != float('inf')
        ])

    if identity_ratios and non_id_ratios:
        u_stat, u_p = stats.mannwhitneyu(
            identity_ratios, non_id_ratios, alternative='greater'
        )
        d_id_vs_nonid = cohens_d(np.array(identity_ratios), np.array(non_id_ratios))

        print(f"\n  Identity vs Non-Identity F/C ratios:")
        print(f"    Identity mean ratio: {np.mean(identity_ratios):.3f}×")
        print(f"    Non-ID mean ratio:   {np.mean(non_id_ratios):.3f}×")
        print(f"    Mann-Whitney p = {u_p:.6e}")
        print(f"    Cohen's d = {d_id_vs_nonid:.4f}")

    # ============================================================
    # Visualization
    # ============================================================
    print("\n📈 Generating figures...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("C1 Control: Is Function-Word Amplification Identity-Specific?",
                 fontsize=13, fontweight='bold', y=1.02)

    cond_names = list(PERTURBATIONS.keys())
    cond_labels = ["Identity\n(Name)", "Number\nSwap", "Adjective\nSwap",
                   "Institution\nSwap", "Rare\nNon-ID"]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # Panel 1: F/C Ratio bar chart
    ratios = [condition_summaries[c]["func_cont_ratio"] for c in cond_names]
    bars = axes[0].bar(range(len(cond_names)), ratios, color=colors, alpha=0.8,
                       edgecolor='white', linewidth=1.5)
    axes[0].axhline(y=1.0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    axes[0].set_xticks(range(len(cond_names)))
    axes[0].set_xticklabels(cond_labels, fontsize=9)
    axes[0].set_ylabel("Function / Content TCD Ratio", fontsize=11)
    axes[0].set_title("Func/Cont TCD Ratio by Perturbation Type", fontsize=11)
    axes[0].grid(alpha=0.2, axis='y')

    for i, (r, bar) in enumerate(zip(ratios, bars)):
        sig = condition_summaries[cond_names[i]]
        marker = "***" if sig["wilcoxon_p"] < 0.001 else "**" if sig["wilcoxon_p"] < 0.01 else "*" if sig["wilcoxon_p"] < 0.05 else ""
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                     f'{r:.2f}×\n{marker}', ha='center', va='bottom', fontsize=9,
                     fontweight='bold')

    # Panel 2: Grouped bar (Func-TCD vs Cont-TCD)
    x = np.arange(len(cond_names))
    width = 0.35
    func_vals = [condition_summaries[c]["mean_func_tcd"] for c in cond_names]
    cont_vals = [condition_summaries[c]["mean_cont_tcd"] for c in cond_names]
    axes[1].bar(x - width/2, func_vals, width, label='Function Words',
                color='#e74c3c', alpha=0.7)
    axes[1].bar(x + width/2, cont_vals, width, label='Content Words',
                color='#3498db', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cond_labels, fontsize=9)
    axes[1].set_ylabel("Mean |TCD|", fontsize=11)
    axes[1].set_title("Func vs Cont TCD by Perturbation Type", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2, axis='y')

    plt.tight_layout()
    fig_path = output_dir / "non_identity_perturbation.pdf"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "non_identity_perturbation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {fig_path}")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        "experiment": "non_identity_perturbation",
        "description": "Tests whether function-word TCD amplification is "
                       "specific to identity swaps (C1)",
        "summaries": condition_summaries,
        "identity_vs_nonid": {
            "identity_mean_ratio": float(np.mean(identity_ratios)) if identity_ratios else 0,
            "nonid_mean_ratio": float(np.mean(non_id_ratios)) if non_id_ratios else 0,
            "mann_whitney_p": float(u_p) if identity_ratios and non_id_ratios else 1.0,
            "cohens_d": float(d_id_vs_nonid) if identity_ratios and non_id_ratios else 0,
        },
    }

    with open(output_dir / "non_identity_perturbation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  ✅ Results saved to {output_dir}")

    # ============================================================
    # Verdict
    # ============================================================
    print(f"\n{'='*70}")
    print("🏁 C1 VERDICT")
    print(f"{'='*70}")

    id_ratio = condition_summaries["IDENTITY"]["func_cont_ratio"]
    max_nonid = max(condition_summaries[c]["func_cont_ratio"]
                    for c in ["NUMBER", "ADJECTIVE", "INSTITUTION", "RARE_NONID"])

    if id_ratio > max_nonid * 1.15:
        print(f"""
  ✅ IDENTITY-SPECIFIC: Identity swaps show higher F/C ratio ({id_ratio:.2f}×)
     than all non-identity perturbations (max: {max_nonid:.2f}×).
     → The function-word effect IS amplified by identity markers.
     → Paper framing is supported.
""")
    elif id_ratio > 1.1 and all(
        condition_summaries[c]["func_cont_ratio"] > 1.0
        for c in PERTURBATIONS
    ):
        print(f"""
  ⚠️ MIXED: All perturbation types show F/C ratio > 1.0.
     Identity: {id_ratio:.2f}× | Max non-ID: {max_nonid:.2f}×
     → Function-word amplification is partially a general MaxSim property.
     → Paper should reframe: "Identity swaps exploit a general function-word
       sensitivity mechanism, but identity markers amplify it."
""")
    else:
        print(f"""
  ❌ NOT IDENTITY-SPECIFIC: Non-identity perturbations show similar or higher
     F/C ratios. Identity: {id_ratio:.2f}× | Max non-ID: {max_nonid:.2f}×
     → Reframe paper: "Function words are inherently more sensitive to
       document perturbations in MaxSim, and identity swaps exploit this."
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Non-identity perturbation experiment (C1 control)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/01_control/non_identity_perturbation",
    )
    args = parser.parse_args()
    run_experiment(args.output_dir)
