#!/usr/bin/env python3
"""Orthogonalized subset panel for the upgraded P2 analysis.

This script uses the raw upgraded P2 results and carves out cleaner matched
subsets so we can test race, gender, frequency, tokenization, and absolute
rarity under tighter controls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "results" / "p2_name_confound" / "name_confound_raw.csv"
OUT_DIR = ROOT / "results" / "p2_name_confound"
SUMMARY_PATH = OUT_DIR / "p2_orthogonalized_panel_summary.json"
REPORT_PATH = OUT_DIR / "p2_orthogonalized_panel_report.txt"
PLOT_PATH = OUT_DIR / "p2_orthogonalized_panel.png"


def fit_clustered(df: pd.DataFrame, formula: str):
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["pair_id"]},
    )
    return model


def extract_term(model, term: str) -> dict:
    return {
        "coef": float(model.params.get(term, 0.0)),
        "stderr": float(model.bse.get(term, 0.0)),
        "pvalue": float(model.pvalues.get(term, 1.0)),
        "significant": bool(model.pvalues.get(term, 1.0) < 0.05),
    }


def build_subsets(df: pd.DataFrame) -> dict:
    return {
        "race_isolated": {
            "description": "Same gender, matched token count, small frequency gap; test cross-race only.",
            "mask": (df["same_gender"] == 1) & (df["token_diff"] == 0) & (df["freq_diff"] <= 0.25),
            "term": "cross_race",
            "formula": "{y} ~ cross_race + C(profession) + C(template_id)",
        },
        "gender_isolated": {
            "description": "Same race, matched token count, small frequency gap; test cross-gender only.",
            "mask": (df["same_race"] == 1) & (df["token_diff"] == 0) & (df["freq_diff"] <= 0.25),
            "term": "cross_gender",
            "formula": "{y} ~ cross_gender + C(profession) + C(template_id)",
        },
        "frequency_gap_isolated": {
            "description": "Same race, same gender, matched token count; test frequency disparity.",
            "mask": (df["same_race"] == 1) & (df["same_gender"] == 1) & (df["token_diff"] == 0),
            "term": "freq_diff",
            "formula": "{y} ~ freq_diff + mean_log_freq + C(profession) + C(template_id)",
        },
        "tokenization_isolated": {
            "description": "Same race, same gender, tightly matched frequency; test token count gap.",
            "mask": (df["same_race"] == 1) & (df["same_gender"] == 1) & (df["freq_diff"] <= 0.25),
            "term": "token_diff",
            "formula": "{y} ~ token_diff + mean_tokens + C(profession) + C(template_id)",
        },
        "absolute_rarity_probe": {
            "description": "Same race, same gender, matched token count, near-equal pair frequency; probe mean rarity.",
            "mask": (df["same_race"] == 1)
            & (df["same_gender"] == 1)
            & (df["token_diff"] == 0)
            & (df["freq_diff"] <= 0.10),
            "term": "mean_log_freq",
            "formula": "{y} ~ mean_log_freq + C(profession) + C(template_id)",
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH)
    subsets = build_subsets(df)

    results = {}
    plot_rows = []

    for subset_name, config in subsets.items():
        subset_df = df.loc[config["mask"]].copy()
        results[subset_name] = {
            "description": config["description"],
            "n_rows": int(len(subset_df)),
            "n_pairs": int(subset_df["pair_id"].nunique()),
            "outcomes": {},
        }

        for outcome in ("ss", "func_tcd"):
            formula = config["formula"].format(y=outcome)
            model = fit_clustered(subset_df, formula)
            term_stats = extract_term(model, config["term"])
            term_stats["r_squared"] = float(model.rsquared)
            results[subset_name]["outcomes"][outcome] = term_stats

            plot_rows.append(
                {
                    "subset": subset_name,
                    "outcome": outcome,
                    "term": config["term"],
                    "coef": term_stats["coef"],
                    "stderr": term_stats["stderr"],
                    "pvalue": term_stats["pvalue"],
                }
            )

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    report_lines = [
        "P2 orthogonalized subset panel",
        "=" * 60,
        "",
    ]
    for subset_name, info in results.items():
        report_lines.append(subset_name)
        report_lines.append(f"- {info['description']}")
        report_lines.append(f"- rows={info['n_rows']}, pairs={info['n_pairs']}")
        for outcome, stats in info["outcomes"].items():
            star = "***" if stats["pvalue"] < 0.001 else "**" if stats["pvalue"] < 0.01 else "*" if stats["pvalue"] < 0.05 else "n.s."
            report_lines.append(
                f"- {outcome}: coef={stats['coef']:+.6f}, p={stats['pvalue']:.4g}, R2={stats['r_squared']:.4f} {star}"
            )
        report_lines.append("")

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    plot_df = pd.DataFrame(plot_rows)
    plot_df["label"] = plot_df["subset"] + "\n" + plot_df["outcome"]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        x=plot_df["coef"],
        y=range(len(plot_df)),
        xerr=1.96 * plot_df["stderr"],
        fmt="o",
        color="#1f4e79",
        ecolor="#7aa6d8",
        capsize=4,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title("P2 Orthogonalized Subset Panel")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close()

    print("✅ Saved orthogonalized P2 subset panel.")


if __name__ == "__main__":
    main()
