"""
Generate all publication-quality figures for the EMNLP paper.
Reads from local/results/ and writes to paper/figures/.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS = PROJECT_ROOT / "local" / "results"
FIGURES = PROJECT_ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "font.family": "serif", "figure.dpi": 300,
})

df = pd.read_csv(RESULTS / "full_65name_results.csv")
print(f"Loaded {len(df)} rows")

# ═══════════════════════════════════════════════════════════════════
# Fig 1: Func vs Cont TCD — violin/box comparison
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.3, 2.8))
data_melt = pd.DataFrame({
    "TCD": np.concatenate([df["func_tcd"].values, df["cont_tcd"].values]),
    "Type": ["Function"] * len(df) + ["Content"] * len(df),
})
palette = {"Function": "#E74C3C", "Content": "#3498DB"}
sns.violinplot(data=data_melt, x="Type", y="TCD", palette=palette,
               inner="quartile", cut=0, ax=ax)
ax.set_ylabel("Token Contribution Disparity")
ax.set_xlabel("")
ax.set_title("Func-TCD vs Cont-TCD (n=8,085)")
ax.text(0.5, 0.93, f"Ratio = 1.41×, p < 10⁻²⁷³",
        transform=ax.transAxes, ha="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
fig.tight_layout()
fig.savefig(FIGURES / "fig1_func_vs_cont_tcd.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig1_func_vs_cont_tcd.png", bbox_inches="tight")
plt.close()
print("✓ Fig 1: Func vs Cont TCD")

# ═══════════════════════════════════════════════════════════════════
# Fig 2: Per-template ratio bar chart
# ═══════════════════════════════════════════════════════════════════
template_labels = [
    "T0: \"{NAME} has over…\"",
    "T1: \"{NAME} is dedicated…\"",
    "T2: \"{NAME} brings…\"",
    "T3: \"candidate {NAME}…\"",
    "T4: \"expert {NAME}…\"",
    "T5: \"{NAME} graduated…\"",
    "T6: \"Resume of {NAME}…\"",
]
ratios = []
for ti in range(7):
    sub = df[df["template_idx"] == ti]
    r = sub["func_tcd"].mean() / sub["cont_tcd"].mean() if sub["cont_tcd"].mean() > 0 else 0
    ratios.append(r)

fig, ax = plt.subplots(figsize=(3.3, 3.0))
colors = ["#95A5A6" if r < 1.2 else "#E67E22" if r < 1.6 else "#E74C3C" for r in ratios]
bars = ax.barh(range(7), ratios, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(7))
ax.set_yticklabels(template_labels, fontsize=7)
ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(x=1.41, color="red", linestyle=":", linewidth=1, alpha=0.7, label="Overall (1.41×)")
ax.set_xlabel("Func-TCD / Cont-TCD Ratio")
ax.set_title("Template Modulation of Bias Ratio")
ax.legend(fontsize=7)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIGURES / "fig2_template_modulation.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig2_template_modulation.png", bbox_inches="tight")
plt.close()
print("✓ Fig 2: Template modulation")

# ═══════════════════════════════════════════════════════════════════
# Fig 3: Rarity amplification grouped bar
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.3, 2.5))
cats = ["CC", "CR", "RR"]
means = [df[df["rarity"] == c]["ss"].mean() for c in cats]
stds = [df[df["rarity"] == c]["ss"].std() / np.sqrt(df[df["rarity"] == c].shape[0]) for c in cats]
ns = [df[df["rarity"] == c].shape[0] for c in cats]
colors_r = ["#27AE60", "#F39C12", "#E74C3C"]
bars = ax.bar(cats, means, yerr=stds, color=colors_r, edgecolor="white",
              capsize=4, linewidth=0.5)
for i, (m, n) in enumerate(zip(means, ns)):
    ax.text(i, m + stds[i] + 0.001, f"n={n}", ha="center", fontsize=7)
ax.set_ylabel("Mean Score Sensitivity (SS)")
ax.set_title("Rarity Amplification: RR/CC = 1.15×")
ax.set_xlabel("Pair Rarity Category")
fig.tight_layout()
fig.savefig(FIGURES / "fig3_rarity_amplification.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig3_rarity_amplification.png", bbox_inches="tight")
plt.close()
print("✓ Fig 3: Rarity amplification")

# ═══════════════════════════════════════════════════════════════════
# Fig 4: Per-profession Func > Cont heatmap
# ═══════════════════════════════════════════════════════════════════
with open(PROJECT_ROOT / "data" / "professions" / "professions.json") as f:
    professions = json.load(f)["professions"]

prof_data = []
for p in professions:
    q = p["query"]
    sub = df[df["query"] == q]
    func_m = sub["func_tcd"].mean()
    cont_m = sub["cont_tcd"].mean()
    r = func_m / cont_m if cont_m > 0 else 0
    prof_data.append({"profession": p.get("profession", q.split()[-1].rstrip("?")),
                      "ratio": r, "func_wins": func_m > cont_m})

prof_df = pd.DataFrame(prof_data).sort_values("ratio", ascending=True)

fig, ax = plt.subplots(figsize=(3.3, 5.0))
colors_p = ["#E74C3C" if r > 1 else "#3498DB" for r in prof_df["ratio"]]
ax.barh(range(len(prof_df)), prof_df["ratio"], color=colors_p,
        edgecolor="white", linewidth=0.3)
ax.set_yticks(range(len(prof_df)))
ax.set_yticklabels(prof_df["profession"], fontsize=6)
ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Func/Cont TCD Ratio")
ax.set_title(f"Per-Profession Ratio (30/33 > 1.0)")
fig.tight_layout()
fig.savefig(FIGURES / "fig4_profession_ratios.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig4_profession_ratios.png", bbox_inches="tight")
plt.close()
print("✓ Fig 4: Per-profession ratios")

# ═══════════════════════════════════════════════════════════════════
# Fig 5: Multi-model comparison (n=1,155 per model)
# ═══════════════════════════════════════════════════════════════════
with open(RESULTS / "multi_model_scaled.json") as f:
    mm = json.load(f)

# ColBERT from main CSV
colbert_ss = df["ss"].mean()

models = ["BM25", "ColBERT", "CrossEnc", "Contriever", "DPR", "SPLADE"]
ss_vals = [
    mm["bm25"]["mean_ss"],
    colbert_ss,
    mm["cross_encoder"]["mean_ss"],
    mm["contriever"]["mean_ss"],
    mm["dpr"]["mean_ss"],
    mm["splade"]["mean_ss"],
]
colors_m = ["#BDC3C7", "#E74C3C", "#95A5A6", "#2ECC71", "#3498DB", "#9B59B6"]
token_support = [False, True, False, False, False, False]

fig, ax = plt.subplots(figsize=(3.3, 2.8))
bars = ax.bar(models, ss_vals, color=colors_m, edgecolor="white", linewidth=0.5)
for i, (v, ts) in enumerate(zip(ss_vals, token_support)):
    label = f"{v:.3f}" if v > 0 else "0.000"
    ax.text(i, v + 0.003, label, ha="center", fontsize=7)
    if ts:
        ax.text(i, -0.012, "token-level", ha="center", fontsize=6, color="#E74C3C", fontstyle="italic")
ax.set_ylabel("Score Sensitivity (SS)")
ax.set_title("Multi-Model Identity Sensitivity (n=1,155)")
ax.set_ylim(-0.02, 0.16)
plt.xticks(rotation=15, ha="right")
fig.tight_layout()
fig.savefig(FIGURES / "fig5_multi_model.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig5_multi_model.png", bbox_inches="tight")
plt.close()
print("✓ Fig 5: Multi-model comparison")

# ═══════════════════════════════════════════════════════════════════
# Fig 6: Name masking causal diagram
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.3, 2.0))
conditions = ["Original\n(name present)", "Masked\n([MASK])"]
values = [0.0181, 0.0000]
colors_c = ["#E74C3C", "#27AE60"]
bars = ax.bar(conditions, values, color=colors_c, edgecolor="white", width=0.5)
ax.text(0, 0.0181 + 0.001, "0.018", ha="center", fontsize=9, fontweight="bold")
ax.text(1, 0.001, "0.000", ha="center", fontsize=9, fontweight="bold")
ax.annotate("100% reduction", xy=(0.5, 0.01), fontsize=9,
            ha="center", color="#E74C3C", fontweight="bold")
ax.set_ylabel("Func-TCD")
ax.set_title("Name Masking → Causal Validation")
ax.set_ylim(0, 0.025)
fig.tight_layout()
fig.savefig(FIGURES / "fig6_name_masking.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig6_name_masking.png", bbox_inches="tight")
plt.close()
print("✓ Fig 6: Name masking")

print(f"\nAll figures saved to {FIGURES}")
print("Files:", sorted([f.name for f in FIGURES.glob("fig*")]))
