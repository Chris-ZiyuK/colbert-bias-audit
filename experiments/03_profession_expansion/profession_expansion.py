"""
P1: Systematic Profession Expansion
=====================================
Expand from 10 → 33 professions across 7 BLS categories.
Analyze: per-profession TCD breakdown, function-word bias ratio,
profession-bias correlation with BLS gender ratio.

PI Feedback #2: "The profession-specific variation in your initial results
is interesting - it might be worth developing this into a more systematic
analysis across a wider range of professions."

Run: conda run -n colbert_bias python profession_expansion.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import warnings
from scipy import stats
from src.audit.core import (
    load_model, encode, get_tokens, maxsim_detail,
    classify_token, get_token_count, compute_tcd_breakdown
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# Configuration
# ============================================================
RESULTS_DIR = PROJECT_ROOT / "results" / "p1_profession_expansion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load profession data
with open(PROJECT_ROOT / "data" / "professions" / "professions.json") as f:
    PROFESSIONS = json.load(f)["professions"]

print(f"📋 Loaded {len(PROFESSIONS)} professions across BLS categories:")
categories = {}
for p in PROFESSIONS:
    categories.setdefault(p["bls_category"], []).append(p["name"])
for cat, profs in categories.items():
    print(f"  {cat}: {', '.join(profs)}")

# Name pairs for counterfactual testing (from existing experiments)
NAME_PAIRS = [
    ("Emily", "Lakisha", "White-F", "Black-F", "race"),
    ("Greg", "Jamal", "White-M", "Black-M", "race"),
    ("Sarah", "Tamika", "White-F", "Black-F", "race"),
    ("Todd", "Darnell", "White-M", "Black-M", "race"),
    ("Anne", "Aisha", "White-F", "Black-F", "race"),
    ("Brett", "Tyrone", "White-M", "Black-M", "race"),
    ("Emily", "Maria", "White-F", "Hispanic-F", "race"),
    ("Greg", "Carlos", "White-M", "Hispanic-M", "race"),
    ("Emily", "Mei", "White-F", "Asian-F", "race"),
    ("Greg", "Wei", "White-M", "Asian-M", "race"),
]

PRONOUN_PAIRS = [
    ("He", "She", "Male", "Female", "gender"),
    ("He", "They", "Male", "Nonbinary", "gender"),
    ("She", "They", "Female", "Nonbinary", "gender"),
]

TEMPLATES = [
    "{name} has extensive experience in this field.",
    "{name} is widely recognized as a leading professional.",
    "{name} has been honored with multiple awards for outstanding work.",
]

PRONOUN_TEMPLATES = [
    "{pron} has extensive experience in this field.",
    "{pron} is widely recognized as a leading professional.",
    "{pron} has been honored with multiple awards for outstanding work.",
]

# ============================================================
# Load Model
# ============================================================
tokenizer, model, device = load_model()

# ============================================================
# Run Experiment
# ============================================================
total_tests = len(PROFESSIONS) * (len(NAME_PAIRS) + len(PRONOUN_PAIRS)) * len(TEMPLATES)
print(f"\n🔬 Total tests: {len(PROFESSIONS)} professions × "
      f"({len(NAME_PAIRS)} name + {len(PRONOUN_PAIRS)} pronoun) × "
      f"{len(TEMPLATES)} templates = {total_tests}")

results = []
test_count = 0

for prof in PROFESSIONS:
    query = prof["query"]
    q_emb = encode(query, tokenizer, model, device, is_query=True)
    q_tokens = get_tokens(query, tokenizer, is_query=True)

    # Name pairs
    for name_a, name_b, label_a, label_b, axis in NAME_PAIRS:
        for template in TEMPLATES:
            doc_a = template.format(name=name_a)
            doc_b = template.format(name=name_b)
            d_emb_a = encode(doc_a, tokenizer, model, device)
            d_emb_b = encode(doc_b, tokenizer, model, device)

            breakdown = compute_tcd_breakdown(q_emb, d_emb_a, d_emb_b, q_tokens)

            results.append({
                "profession": prof["name"],
                "bls_category": prof["bls_category"],
                "prestige": prof["prestige"],
                "bls_female_pct": prof["bls_female_pct"],
                "name_a": name_a, "name_b": name_b,
                "label_a": label_a, "label_b": label_b,
                "axis": axis, "swap_type": "name",
                "ss": breakdown["ss"],
                "func_tcd": breakdown["func_tcd"],
                "cont_tcd": breakdown["cont_tcd"],
                "tcd_ratio": breakdown["tcd_ratio"],
            })
            test_count += 1

    # Pronoun pairs
    for pron_a, pron_b, label_a, label_b, axis in PRONOUN_PAIRS:
        for template in PRONOUN_TEMPLATES:
            doc_a = template.format(pron=pron_a)
            doc_b = template.format(pron=pron_b)
            d_emb_a = encode(doc_a, tokenizer, model, device)
            d_emb_b = encode(doc_b, tokenizer, model, device)

            breakdown = compute_tcd_breakdown(q_emb, d_emb_a, d_emb_b, q_tokens)

            results.append({
                "profession": prof["name"],
                "bls_category": prof["bls_category"],
                "prestige": prof["prestige"],
                "bls_female_pct": prof["bls_female_pct"],
                "name_a": pron_a, "name_b": pron_b,
                "label_a": label_a, "label_b": label_b,
                "axis": axis, "swap_type": "pronoun",
                "ss": breakdown["ss"],
                "func_tcd": breakdown["func_tcd"],
                "cont_tcd": breakdown["cont_tcd"],
                "tcd_ratio": breakdown["tcd_ratio"],
            })
            test_count += 1

    print(f"  ✅ {prof['name']:25s} ({test_count}/{total_tests})")

df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR / "profession_expansion_raw.csv", index=False)
print(f"\n💾 Saved {len(df)} rows to profession_expansion_raw.csv")

# ============================================================
# Analysis 1: Per-Profession Bias Summary
# ============================================================
print("\n" + "=" * 70)
print("📊 ANALYSIS 1: Per-Profession Bias Summary (33 professions)")
print("=" * 70)

prof_summary = df.groupby(["profession", "bls_category", "prestige", "bls_female_pct"]).agg(
    mean_ss=("ss", "mean"),
    mean_func_tcd=("func_tcd", "mean"),
    mean_cont_tcd=("cont_tcd", "mean"),
    mean_tcd_ratio=("tcd_ratio", "mean"),
    n_tests=("ss", "count"),
).reset_index().sort_values("mean_ss", ascending=False)

print(f"\n{'Profession':>25s} {'Category':>12s} {'Female%':>8s} {'Mean SS':>10s} {'Func TCD':>10s} {'Cont TCD':>10s} {'F/C Ratio':>10s}")
print("-" * 90)
for _, row in prof_summary.iterrows():
    func_gt = "✅" if row["mean_func_tcd"] > row["mean_cont_tcd"] else "❌"
    print(f"{row['profession']:>25s} {row['bls_category']:>12s} {row['bls_female_pct']:>7.1f}% "
          f"{row['mean_ss']:>10.6f} {row['mean_func_tcd']:>10.6f} {row['mean_cont_tcd']:>10.6f} "
          f"{row['mean_tcd_ratio']:>9.2f}x {func_gt}")

func_winner_count = sum(1 for _, r in prof_summary.iterrows() if r["mean_func_tcd"] > r["mean_cont_tcd"])
print(f"\nFunction > Content in {func_winner_count}/{len(prof_summary)} professions")

# ============================================================
# Analysis 2: BLS Category Comparison
# ============================================================
print("\n" + "=" * 70)
print("📊 ANALYSIS 2: Bias by BLS Category")
print("=" * 70)

cat_summary = df.groupby("bls_category").agg(
    mean_ss=("ss", "mean"),
    mean_func_tcd=("func_tcd", "mean"),
    n_professions=("profession", "nunique"),
).reset_index().sort_values("mean_ss", ascending=False)

for _, row in cat_summary.iterrows():
    print(f"  {row['bls_category']:>15s}: SS={row['mean_ss']:.6f}, Func-TCD={row['mean_func_tcd']:.6f} "
          f"({row['n_professions']} professions)")

# ============================================================
# Analysis 3: Correlation with BLS Female %
# ============================================================
print("\n" + "=" * 70)
print("📊 ANALYSIS 3: Bias vs BLS Gender Ratio Correlation")
print("=" * 70)

# Pearson correlation
corr_ss, p_ss = stats.pearsonr(prof_summary["bls_female_pct"], prof_summary["mean_ss"])
corr_func, p_func = stats.pearsonr(prof_summary["bls_female_pct"], prof_summary["mean_func_tcd"])

print(f"  Pearson r (Female% vs Mean SS):       r={corr_ss:+.4f}, p={p_ss:.6f} {'***' if p_ss < 0.001 else '**' if p_ss < 0.01 else '*' if p_ss < 0.05 else 'n.s.'}")
print(f"  Pearson r (Female% vs Mean Func-TCD): r={corr_func:+.4f}, p={p_func:.6f} {'***' if p_func < 0.001 else '**' if p_func < 0.01 else '*' if p_func < 0.05 else 'n.s.'}")

# ============================================================
# Analysis 4: Prestige Level Comparison
# ============================================================
print("\n" + "=" * 70)
print("📊 ANALYSIS 4: Bias by Prestige Level")
print("=" * 70)

prest_summary = df.groupby("prestige").agg(
    mean_ss=("ss", "mean"),
    mean_func_tcd=("func_tcd", "mean"),
).reset_index()

for _, row in prest_summary.iterrows():
    print(f"  {row['prestige']:>6s}: SS={row['mean_ss']:.6f}, Func-TCD={row['mean_func_tcd']:.6f}")

# Kruskal-Wallis test
groups = [g["ss"].values for _, g in df.groupby("prestige")]
h_stat, kw_p = stats.kruskal(*groups)
print(f"\n  Kruskal-Wallis (SS across prestige levels): H={h_stat:.2f}, p={kw_p:.6f}")

# ============================================================
# Visualizations
# ============================================================
print("\n📈 Generating visualizations...")

# Set style
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white'
})

# --- Fig 1: Profession Bias Ranking ---
fig, ax = plt.subplots(figsize=(12, 10))
colors_map = {
    "healthcare": "#e74c3c", "STEM": "#3498db", "education": "#2ecc71",
    "white-collar": "#9b59b6", "leadership": "#e67e22", "blue-collar": "#34495e",
    "service": "#1abc9c"
}
bar_colors = [colors_map.get(row["bls_category"], "#888") for _, row in prof_summary.iterrows()]

ax.barh(range(len(prof_summary)), prof_summary["mean_ss"].values,
        color=bar_colors, alpha=0.8, edgecolor='white')
ax.set_yticks(range(len(prof_summary)))
ax.set_yticklabels(prof_summary["profession"].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Mean Score Sensitivity (SS)")
ax.set_title("Bias Magnitude by Profession (33 Professions)\nColored by BLS Category")
ax.grid(axis='x', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=cat) for cat, c in colors_map.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "profession_bias_ranking.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ profession_bias_ranking.png")

# --- Fig 2: Function/Content Ratio per Profession ---
fig, ax = plt.subplots(figsize=(12, 10))
ratios = prof_summary["mean_tcd_ratio"].values
bar_colors_ratio = ['#27ae60' if r > 1 else '#e67e22' for r in ratios]

ax.barh(range(len(prof_summary)), ratios, color=bar_colors_ratio, alpha=0.8, edgecolor='white')
ax.set_yticks(range(len(prof_summary)))
ax.set_yticklabels(prof_summary["profession"].values, fontsize=9)
ax.axvline(x=1.0, color='black', linewidth=1.5, linestyle='--', label='Equal (1.0)')
ax.invert_yaxis()
ax.set_xlabel("Function-word TCD / Content-word TCD")
ax.set_title("Function-word vs Content-word Bias Ratio\nGreen = Function words carry more bias")
ax.grid(axis='x', alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "func_content_ratio.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ func_content_ratio.png")

# --- Fig 3: Bias vs BLS Female % Scatter ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_col, y_label, title in [
    (axes[0], "mean_ss", "Mean Score Sensitivity", f"SS vs Female% (r={corr_ss:+.3f})"),
    (axes[1], "mean_func_tcd", "Mean Function-word TCD", f"Func-TCD vs Female% (r={corr_func:+.3f})"),
]:
    scatter_colors = [colors_map.get(row["bls_category"], "#888") for _, row in prof_summary.iterrows()]
    ax.scatter(prof_summary["bls_female_pct"], prof_summary[y_col],
               c=scatter_colors, s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
    # Add profession labels
    for _, row in prof_summary.iterrows():
        ax.annotate(row["profession"], (row["bls_female_pct"], row[y_col]),
                    fontsize=6, alpha=0.7, ha='left')
    # Regression line
    z = np.polyfit(prof_summary["bls_female_pct"], prof_summary[y_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(prof_summary["bls_female_pct"].min(), prof_summary["bls_female_pct"].max(), 100)
    ax.plot(x_line, p(x_line), '--', color='red', alpha=0.5, linewidth=1.5)
    ax.set_xlabel("BLS Female Percentage (%)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "bias_vs_gender_ratio.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ bias_vs_gender_ratio.png")

# --- Fig 4: Category Heatmap ---
fig, ax = plt.subplots(figsize=(10, 6))
pivot = df.groupby(["bls_category", "axis"])["ss"].mean().unstack(fill_value=0)
sns.heatmap(pivot, annot=True, fmt=".5f", cmap="Reds", ax=ax, linewidths=0.5)
ax.set_title("Mean Score Sensitivity: BLS Category × Bias Axis")
ax.set_ylabel("BLS Category")
ax.set_xlabel("Bias Axis")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "category_axis_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ category_axis_heatmap.png")

# ============================================================
# Save Summary
# ============================================================
summary = {
    "total_tests": test_count,
    "n_professions": len(PROFESSIONS),
    "n_categories": len(categories),
    "func_gt_content_count": int(func_winner_count),
    "func_gt_content_total": len(prof_summary),
    "correlation_female_pct_vs_ss": {"r": float(corr_ss), "p": float(p_ss)},
    "correlation_female_pct_vs_func_tcd": {"r": float(corr_func), "p": float(p_func)},
    "per_category": cat_summary.set_index("bls_category").to_dict(orient="index"),
    "per_profession": prof_summary.set_index("profession").to_dict(orient="index"),
}

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

prof_summary.to_csv(RESULTS_DIR / "profession_summary.csv", index=False)

print(f"\n💾 All results saved to: {RESULTS_DIR}/")
print(f"\n✨ P1 Profession Expansion DONE! {test_count} tests across {len(PROFESSIONS)} professions.")
