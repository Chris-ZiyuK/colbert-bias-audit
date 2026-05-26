"""
Multiple Comparison Correction — Addressing Reviewer D5
========================================================
Applies Benjamini-Hochberg FDR and Bonferroni corrections
to all 33 professions × 7 templates = 231 individual tests.
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from src.audit.core import (load_model, encode, get_tokens, maxsim_detail,
                            classify_token, compute_tcd_breakdown)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME_PAIRS = [("Emily","Lakisha"),("Brett","Jamal"),("Sarah","Tyrone"),
              ("Allison","Tanisha"),("Neil","Leroy")]

PROFESSIONS = ["doctor","nurse","surgeon","therapist","dentist",
               "software engineer","researcher","data scientist","civil engineer","biologist",
               "teacher","professor","librarian","lawyer","accountant",
               "financial analyst","consultant","CEO","manager","director",
               "electrician","plumber","mechanic","construction worker","truck driver",
               "welder","secretary","social worker","receptionist","janitor",
               "chef","pilot","firefighter"]

TEMPLATES = [
    "{NAME} has over ten years of experience in this field and has worked with top organizations.",
    "{NAME} is a dedicated professional with extensive expertise and a proven track record.",
    "{NAME} brings extensive experience and a unique perspective to every project undertaken.",
    "The candidate {NAME} holds advanced credentials and has received multiple awards.",
    "As a leading expert, {NAME} has published widely and contributed to major initiatives.",
    "{NAME} graduated from a top university and has been recognized for outstanding work.",
    "Resume of {NAME}, an experienced professional with deep domain expertise.",
]

def run_experiment(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer, model, device = load_model()

    all_pvalues = []
    all_effects = []
    all_labels = []
    test_count = 0

    for p_idx, prof in enumerate(PROFESSIONS):
        query = f"Who is an experienced {prof}?"
        q_emb = encode(query, tokenizer, model, device, is_query=True)
        q_tokens = get_tokens(query, tokenizer, is_query=True)

        for t_idx, tmpl in enumerate(TEMPLATES):
            func_tcds, cont_tcds = [], []

            for name_a, name_b in NAME_PAIRS:
                doc_a = tmpl.replace("{NAME}", name_a)
                doc_b = tmpl.replace("{NAME}", name_b)
                d_emb_a = encode(doc_a, tokenizer, model, device, is_query=False)
                d_emb_b = encode(doc_b, tokenizer, model, device, is_query=False)

                result = compute_tcd_breakdown(q_emb, d_emb_a, d_emb_b, q_tokens)
                if result["func_tcd"] is not None:
                    func_tcds.append(result["func_tcd"])
                if result["cont_tcd"] is not None:
                    cont_tcds.append(result["cont_tcd"])

            # Test: are func_tcds significantly > cont_tcds?
            if len(func_tcds) >= 3:
                diffs = [f - c for f, c in zip(func_tcds, cont_tcds)]
                try:
                    stat, p = stats.wilcoxon(diffs, alternative="greater")
                except ValueError:
                    p = 1.0
                d = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0
            else:
                p = 1.0
                d = 0

            all_pvalues.append(p)
            all_effects.append(d)
            all_labels.append(f"{prof} × T{t_idx}")
            test_count += 1

        if (p_idx + 1) % 10 == 0:
            print(f"  ... {p_idx + 1}/{len(PROFESSIONS)} professions ({test_count} tests)")

    print(f"\n✅ {test_count} total tests\n")

    # Apply corrections
    pvals = np.array(all_pvalues)
    alpha = 0.05

    # Uncorrected
    n_sig_uncorrected = np.sum(pvals < alpha)

    # BH-FDR
    reject_bh, pvals_bh, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    n_sig_bh = np.sum(reject_bh)

    # Bonferroni
    reject_bonf, pvals_bonf, _, _ = multipletests(pvals, alpha=alpha, method="bonferroni")
    n_sig_bonf = np.sum(reject_bonf)

    # Holm-Bonferroni (less conservative)
    reject_holm, pvals_holm, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    n_sig_holm = np.sum(reject_holm)

    print("=" * 70)
    print("📊 MULTIPLE COMPARISON CORRECTION (D5)")
    print("=" * 70)
    print(f"\n  Total tests: {test_count} (33 professions × 7 templates)")
    print(f"\n  Correction Method       Significant (p < 0.05)    % of tests")
    print(f"  ──────────────────── ────────────────────── ────────────")
    print(f"  Uncorrected               {n_sig_uncorrected:>4} / {test_count}              {100*n_sig_uncorrected/test_count:.1f}%")
    print(f"  BH-FDR                    {n_sig_bh:>4} / {test_count}              {100*n_sig_bh/test_count:.1f}%")
    print(f"  Holm-Bonferroni           {n_sig_holm:>4} / {test_count}              {100*n_sig_holm/test_count:.1f}%")
    print(f"  Bonferroni                {n_sig_bonf:>4} / {test_count}              {100*n_sig_bonf/test_count:.1f}%")

    # Median effect size among significant (BH-FDR) tests
    sig_effects = [e for e, r in zip(all_effects, reject_bh) if r]
    if sig_effects:
        print(f"\n  Median effect size (BH-significant): d = {np.median(sig_effects):.3f}")
        print(f"  Mean effect size (BH-significant):   d = {np.mean(sig_effects):.3f}")

    # Per-profession summary
    print(f"\n  📊 Per-profession (7-template family, BH-FDR):")
    for p_idx, prof in enumerate(PROFESSIONS):
        start = p_idx * 7
        end = start + 7
        prof_reject = reject_bh[start:end]
        n_sig = np.sum(prof_reject)
        print(f"    {prof:<25} {n_sig}/7 significant")

    # Generate figure
    print("\n📈 Generating figures...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # P-value distribution
    ax = axes[0]
    ax.hist(pvals, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    ax.axvline(x=0.05, color="#e74c3c", linestyle="--", linewidth=2, label="α = 0.05")
    ax.set_xlabel("Uncorrected p-value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"P-value Distribution ({n_sig_uncorrected}/{test_count} sig.)", fontsize=12)
    ax.legend()

    # Correction comparison
    ax = axes[1]
    methods = ["Uncorrected", "BH-FDR", "Holm", "Bonferroni"]
    counts = [n_sig_uncorrected, n_sig_bh, n_sig_holm, n_sig_bonf]
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar(methods, counts, color=colors, alpha=0.85, edgecolor="white", linewidth=2)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{c}", ha="center", fontsize=12, fontweight="bold")
    ax.axhline(y=test_count, color="gray", linestyle=":", alpha=0.5, label=f"Total = {test_count}")
    ax.set_ylabel(f"# Significant Tests (of {test_count})", fontsize=11)
    ax.set_title("Effect of Multiple Comparison Correction", fontsize=12)
    ax.set_ylim(0, test_count + 20)
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fdr_correction.pdf"), dpi=150, bbox_inches="tight")
    print(f"  ✅ fdr_correction.pdf")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({
            "total_tests": test_count,
            "n_sig_uncorrected": int(n_sig_uncorrected),
            "n_sig_bh_fdr": int(n_sig_bh),
            "n_sig_holm": int(n_sig_holm),
            "n_sig_bonferroni": int(n_sig_bonf),
            "median_effect_sig": float(np.median(sig_effects)) if sig_effects else None,
        }, f, indent=2)

    print(f"  ✅ Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/13_multiple_comparison")
    args = parser.parse_args()
    run_experiment(args.output_dir)
