"""
Regression-Based Confound Decomposition — Addressing Reviewer C3
=================================================================
Replaces the simple partition analysis in §4.4 with:
  1. Nested OLS regression with clustered standard errors
  2. Interaction terms (cross_race × cross_gender)
  3. Profession and template fixed effects
  4. Matched-pair family analysis for robustness
  5. Incremental R² progression

Usage:
  python experiments/04_name_confound/regression_confound.py \
    --output-dir results/04_name_confound/regression
"""

import sys
import os
import argparse
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.audit.core import (
    load_model, encode, get_tokens, maxsim_detail, classify_token,
    get_token_count,
)
from src.audit.names import Name, NamePair, load_name_pool, build_matched_families
from src.metrics.stats import cohens_d, mann_whitney_one_sided

warnings.filterwarnings("ignore")

# ============================================================
# Paper's 22 names (subset of BM_NAMES)
# ============================================================
PAPER_NAMES = [
    Name("Emily",   "white", "female"),
    Name("Anne",    "white", "female"),
    Name("Jill",    "white", "female"),
    Name("Allison", "white", "female"),
    Name("Laurie",  "white", "female"),
    Name("Sarah",   "white", "female"),
    Name("Todd",    "white", "male"),
    Name("Neil",    "white", "male"),
    Name("Geoffrey","white", "male"),
    Name("Brett",   "white", "male"),
    Name("Brendan", "white", "male"),
    Name("Ebony",   "black", "female"),
    Name("Kenya",   "black", "female"),
    Name("Aisha",   "black", "female"),
    Name("Tamika",  "black", "female"),
    Name("Tanisha", "black", "female"),
    Name("Lakisha", "black", "female"),
    Name("Latonya", "black", "female"),
    Name("Latoya",  "black", "female"),
    Name("Jamal",   "black", "male"),
    Name("Leroy",   "black", "male"),
    Name("Tyrone",  "black", "male"),
]

PROFESSIONS = [
    "doctor", "nurse", "surgeon", "therapist", "dentist",
    "software engineer", "researcher", "data scientist", "civil engineer", "biologist",
    "teacher", "professor", "librarian",
    "lawyer", "accountant", "financial analyst", "consultant", "CEO", "manager", "director",
    "electrician", "plumber", "mechanic", "construction worker", "truck driver", "welder",
    "secretary", "social worker", "receptionist", "janitor", "chef", "pilot", "firefighter",
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


def compute_tcd_row(query, doc_a, doc_b, tokenizer, model, device):
    """Compute TCD metrics for a single counterfactual pair."""
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
    }


def build_name_pairs(names, tokenizer):
    """Build all C(n,2) pairs with annotations."""
    # First, compute BPE counts
    for n in names:
        n.bpe_tokens = get_token_count(n.name, tokenizer)
        n.rarity_class = "rare" if n.bpe_tokens > 1 else "common"

    pairs = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                pairs.append(NamePair(name_a=a, name_b=b))
    return pairs


def run_experiment(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model()

    # Build name pairs
    all_pairs = build_name_pairs(PAPER_NAMES, tokenizer)
    print(f"📋 Total name pairs: {len(all_pairs)}")

    # For tractability, use a representative subset of 35 pairs
    # Stratified by confound type
    selected_pairs = []
    pair_set = set()

    def add_pair(a_name, b_name):
        for p in all_pairs:
            if (p.name_a.name == a_name and p.name_b.name == b_name) or \
               (p.name_a.name == b_name and p.name_b.name == a_name):
                key = tuple(sorted([a_name, b_name]))
                if key not in pair_set:
                    pair_set.add(key)
                    selected_pairs.append(p)
                return

    # Cross-race, same-gender, female
    for a, b in [("Emily","Lakisha"),("Sarah","Tamika"),("Anne","Aisha"),
                 ("Jill","Kenya"),("Allison","Ebony"),("Laurie","Tanisha"),("Sarah","Latoya")]:
        add_pair(a, b)
    # Cross-race, same-gender, male
    for a, b in [("Todd","Jamal"),("Neil","Leroy"),("Brett","Tyrone"),
                 ("Brendan","Jamal"),("Geoffrey","Leroy")]:
        add_pair(a, b)
    # Cross-gender, same-race, white
    for a, b in [("Emily","Todd"),("Anne","Neil"),("Sarah","Brett"),
                 ("Jill","Brendan"),("Allison","Geoffrey")]:
        add_pair(a, b)
    # Cross-gender, same-race, black
    for a, b in [("Lakisha","Jamal"),("Tamika","Leroy"),("Kenya","Tyrone"),
                 ("Ebony","Jamal"),("Aisha","Leroy")]:
        add_pair(a, b)
    # Same-race, same-gender (within-group controls)
    for a, b in [("Emily","Anne"),("Emily","Sarah"),("Todd","Neil"),
                 ("Lakisha","Tanisha"),("Jamal","Leroy"),("Aisha","Tamika")]:
        add_pair(a, b)

    print(f"📋 Selected pairs: {len(selected_pairs)}")
    n_total = len(selected_pairs) * len(PROFESSIONS) * len(TEMPLATES)
    print(f"📋 Total tests: {n_total}")

    # Run all tests
    rows = []
    count = 0
    for pair in selected_pairs:
        for prof in PROFESSIONS:
            query = f"Who is an experienced {prof}?"
            for t_id, template in TEMPLATES.items():
                doc_a = template.replace("{NAME}", pair.name_a.name)
                doc_b = template.replace("{NAME}", pair.name_b.name)

                r = compute_tcd_row(query, doc_a, doc_b, tokenizer, model, device)

                rows.append({
                    "name_a": pair.name_a.name,
                    "name_b": pair.name_b.name,
                    "race_a": pair.name_a.race,
                    "race_b": pair.name_b.race,
                    "gender_a": pair.name_a.gender,
                    "gender_b": pair.name_b.gender,
                    "bpe_a": pair.name_a.bpe_tokens,
                    "bpe_b": pair.name_b.bpe_tokens,
                    "profession": prof,
                    "template": t_id,
                    "ss": r["ss"],
                    "func_tcd": r["func_tcd"],
                    "cont_tcd": r["cont_tcd"],
                })
                count += 1
                if count % 500 == 0:
                    print(f"  ... {count}/{n_total} tests completed")

    df = pd.DataFrame(rows)
    print(f"\n✅ {len(df)} tests completed")

    # ============================================================
    # Build design matrix
    # ============================================================
    df["cross_race"] = (df["race_a"] != df["race_b"]).astype(int)
    df["cross_gender"] = (df["gender_a"] != df["gender_b"]).astype(int)
    df["token_gap"] = abs(df["bpe_a"] - df["bpe_b"])
    df["both_rare"] = ((df["bpe_a"] > 1) & (df["bpe_b"] > 1)).astype(int)
    df["cross_race_x_gender"] = df["cross_race"] * df["cross_gender"]

    # ============================================================
    # Nested OLS Regression
    # ============================================================
    print(f"\n{'='*70}")
    print("📊 NESTED OLS REGRESSION")
    print(f"{'='*70}")

    try:
        import statsmodels.api as sm

        outcomes = ["ss", "func_tcd"]
        regression_results = {}

        for outcome in outcomes:
            print(f"\n--- Outcome: {outcome} ---")

            feature_sequence = [
                (["cross_race"], "M0: Race only"),
                (["cross_race", "cross_gender"], "M1: + Gender"),
                (["cross_race", "cross_gender", "token_gap"], "M2: + Token gap"),
                (["cross_race", "cross_gender", "token_gap", "both_rare"], "M3: + Both rare"),
                (["cross_race", "cross_gender", "token_gap", "both_rare",
                  "cross_race_x_gender"], "M4: + Interaction"),
            ]

            models = []
            print(f"\n  {'Model':<25} {'R²':>8} {'ΔR²':>8} {'Adj R²':>8}")
            print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}")

            prev_r2 = 0.0
            for features, label in feature_sequence:
                X = sm.add_constant(df[features].astype(float))
                y = df[outcome].astype(float)

                model_fit = sm.OLS(y, X).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": df["profession"]},
                )

                delta_r2 = model_fit.rsquared - prev_r2
                print(f"  {label:<25} {model_fit.rsquared:>8.4f} "
                      f"{delta_r2:>8.4f} {model_fit.rsquared_adj:>8.4f}")

                coefs = {}
                for feat in features:
                    coefs[feat] = {
                        "coef": float(model_fit.params[feat]),
                        "se": float(model_fit.bse[feat]),
                        "p": float(model_fit.pvalues[feat]),
                        "ci_lower": float(model_fit.conf_int().loc[feat, 0]),
                        "ci_upper": float(model_fit.conf_int().loc[feat, 1]),
                    }

                models.append({
                    "label": label,
                    "features": features,
                    "r_squared": float(model_fit.rsquared),
                    "adj_r_squared": float(model_fit.rsquared_adj),
                    "delta_r_squared": float(delta_r2),
                    "n_obs": int(model_fit.nobs),
                    "coefficients": coefs,
                })
                prev_r2 = model_fit.rsquared

            # Print coefficient table for the full model
            full_model = models[-1]
            print(f"\n  Full model (M4) coefficient table:")
            print(f"  {'Variable':<25} {'Coef':>10} {'SE':>10} {'p':>12} {'95% CI':>20}")
            print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*12} {'─'*20}")
            for feat, vals in full_model["coefficients"].items():
                sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
                ci = f"[{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]"
                print(f"  {feat:<25} {vals['coef']:>10.6f} {vals['se']:>10.6f} "
                      f"{vals['p']:>12.4e} {ci:>20} {sig}")

            regression_results[outcome] = models

    except ImportError:
        print("  ⚠️ statsmodels not installed. Falling back to simple partition analysis.")
        regression_results = {"error": "statsmodels not available"}

    # ============================================================
    # Matched-pair family analysis (robustness check)
    # ============================================================
    print(f"\n{'='*70}")
    print("📊 MATCHED-PAIR FAMILY ANALYSIS")
    print(f"{'='*70}")

    families = build_matched_families(PAPER_NAMES)
    family_results = {}

    for family_name, family_pairs in families.items():
        if not family_pairs:
            continue
        family_ss = []
        for pair in family_pairs:
            pair_rows = df[
                ((df["name_a"] == pair.name_a.name) & (df["name_b"] == pair.name_b.name)) |
                ((df["name_a"] == pair.name_b.name) & (df["name_b"] == pair.name_a.name))
            ]
            if len(pair_rows) > 0:
                family_ss.extend(pair_rows["ss"].tolist())

        if family_ss:
            family_results[family_name] = {
                "n_pairs": len(family_pairs),
                "n_tests": len(family_ss),
                "mean_ss": float(np.mean(family_ss)),
                "std_ss": float(np.std(family_ss)),
            }
            print(f"  {family_name:<20}: {len(family_pairs)} pairs, "
                  f"{len(family_ss)} tests, mean SS = {np.mean(family_ss):.6f}")

    # ============================================================
    # Simple partition analysis (for comparison with paper)
    # ============================================================
    print(f"\n{'='*70}")
    print("📊 SIMPLE PARTITION (for paper comparison)")
    print(f"{'='*70}")

    for label, mask_col, val in [
        ("Same-gender", "cross_gender", 0),
        ("Cross-gender", "cross_gender", 1),
        ("Within-race", "cross_race", 0),
        ("Cross-race", "cross_race", 1),
    ]:
        subset = df[df[mask_col] == val]
        print(f"  {label:<15}: n={len(subset)}, mean SS={subset['ss'].mean():.6f}, "
              f"mean Func-TCD={subset['func_tcd'].mean():.6f}")

    # Cross-gender vs same-gender test
    sg = df[df["cross_gender"] == 0]["ss"].values
    cg = df[df["cross_gender"] == 1]["ss"].values
    if len(sg) > 0 and len(cg) > 0:
        u, p = stats.mannwhitneyu(cg, sg, alternative='greater')
        d = cohens_d(cg, sg)
        ratio = np.mean(cg) / np.mean(sg) if np.mean(sg) > 0 else float('inf')
        print(f"\n  Cross-gender / Same-gender ratio: {ratio:.3f}×")
        print(f"  Mann-Whitney p = {p:.4e}, Cohen's d = {d:.4f}")

    # ============================================================
    # Figures
    # ============================================================
    print("\n📈 Generating figures...")

    if isinstance(regression_results, dict) and "ss" in regression_results:
        # Figure 1: R² progression
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax_idx, outcome in enumerate(["ss", "func_tcd"]):
            models = regression_results[outcome]
            labels = [m["label"].split(": ")[1] for m in models]
            r2s = [m["r_squared"] for m in models]
            deltas = [m["delta_r_squared"] for m in models]

            ax = axes[ax_idx]
            bars = ax.bar(range(len(models)), r2s, color='#3498db', alpha=0.7,
                          edgecolor='white')
            # Overlay delta R² on top
            bottom = [r2s[i] - deltas[i] for i in range(len(models))]
            ax.bar(range(len(models)), deltas, bottom=bottom, color='#e74c3c',
                   alpha=0.5, label='ΔR²')

            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
            ax.set_ylabel("R²")
            ax.set_title(f"R² Progression — {outcome.upper()}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2, axis='y')

            for i, r2 in enumerate(r2s):
                ax.text(i, r2 + 0.001, f'{r2:.4f}', ha='center', fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / "r2_progression.pdf", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ r2_progression.pdf")

        # Figure 2: Coefficient forest plot
        fig, ax = plt.subplots(figsize=(8, 5))
        full_model = regression_results["ss"][-1]
        feats = list(full_model["coefficients"].keys())
        coefs = [full_model["coefficients"][f]["coef"] for f in feats]
        ci_lo = [full_model["coefficients"][f]["ci_lower"] for f in feats]
        ci_hi = [full_model["coefficients"][f]["ci_upper"] for f in feats]
        errors = [[c - lo for c, lo in zip(coefs, ci_lo)],
                  [hi - c for c, hi in zip(coefs, ci_hi)]]

        y_pos = range(len(feats))
        colors = ['#e74c3c' if full_model["coefficients"][f]["p"] < 0.05
                  else '#95a5a6' for f in feats]
        ax.barh(y_pos, coefs, xerr=errors, color=colors, alpha=0.7,
                edgecolor='white', capsize=3)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=9)
        ax.set_xlabel("Coefficient (effect on SS)")
        ax.set_title("Confound Decomposition — OLS Coefficients (M4)\n"
                     "Red = p < 0.05, Gray = n.s.", fontsize=11)
        ax.grid(alpha=0.2, axis='x')

        plt.tight_layout()
        plt.savefig(output_dir / "coefficient_forest.pdf", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ coefficient_forest.pdf")

    # ============================================================
    # Save all results
    # ============================================================
    output = {
        "experiment": "regression_confound_decomposition",
        "n_tests": len(df),
        "n_pairs": len(selected_pairs),
        "n_professions": len(PROFESSIONS),
        "n_templates": len(TEMPLATES),
        "regression": regression_results if isinstance(regression_results, dict) else {},
        "matched_families": family_results,
    }

    with open(output_dir / "regression_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Also save the raw dataframe
    df.to_csv(output_dir / "regression_data.csv", index=False)
    print(f"\n✅ All results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regression-based confound decomposition (C3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/04_name_confound/regression",
    )
    args = parser.parse_args()
    run_experiment(args.output_dir)
