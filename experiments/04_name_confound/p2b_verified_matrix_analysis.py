#!/usr/bin/env python3
"""Analyze the tokenizer-verified P2b pair matrix.

This script reuses the already-computed upgraded P2 raw results and focuses on
the tokenizer-verified treatment/control families exported by
`build_p2b_verified_matrix.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "results" / "p2_name_confound" / "name_confound_raw.csv"
MATRIX_PATH = ROOT / "data" / "audit_names" / "p2b_verified_matrix.json"
OUT_DIR = ROOT / "results" / "p2_name_confound"
SUMMARY_PATH = OUT_DIR / "p2b_verified_matrix_summary.json"
REPORT_PATH = OUT_DIR / "p2b_verified_matrix_report.txt"
PLOT_PATH = OUT_DIR / "p2b_verified_matrix_effects.png"


def fit_clustered(data: pd.DataFrame, formula: str):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["pair_id"]},
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)

    with MATRIX_PATH.open() as f:
        matrix = json.load(f)["families"]

    results = {}
    plot_rows = []
    report_lines = [
        "P2b tokenizer-verified matrix analysis",
        "=" * 60,
        "",
    ]

    for family_name, family in matrix.items():
        pair_df = pd.DataFrame(family["pairs"])
        if pair_df.empty:
            continue

        subset = raw.merge(
            pair_df[["pair_id", "treatment", "group"]],
            on="pair_id",
            how="inner",
        )
        family_summary = {
            "description": family["description"],
            "n_pairs": int(pair_df["pair_id"].nunique()),
            "n_rows": int(len(subset)),
            "n_treatment_pairs": int((pair_df["treatment"] == 1).sum()),
            "n_control_pairs": int((pair_df["treatment"] == 0).sum()),
            "mean_by_group": (
                subset.groupby("group")[["ss", "func_tcd", "cont_tcd"]]
                .mean()
                .round(6)
                .reset_index()
                .to_dict(orient="records")
            ),
            "models": {},
        }

        report_lines.append(family_name)
        report_lines.append(f"- {family['description']}")
        report_lines.append(
            f"- rows={len(subset)}, pairs={pair_df['pair_id'].nunique()}, "
            f"treatment_pairs={int((pair_df['treatment'] == 1).sum())}, "
            f"control_pairs={int((pair_df['treatment'] == 0).sum())}"
        )

        for outcome in ("ss", "func_tcd"):
            model = fit_clustered(subset, f"{outcome} ~ treatment + C(profession) + C(template_id)")
            coef = float(model.params.get("treatment", 0.0))
            stderr = float(model.bse.get("treatment", 0.0))
            pvalue = float(model.pvalues.get("treatment", 1.0))
            family_summary["models"][outcome] = {
                "coef": coef,
                "stderr": stderr,
                "pvalue": pvalue,
                "r_squared": float(model.rsquared),
                "significant": bool(pvalue < 0.05),
            }
            plot_rows.append(
                {
                    "family": family_name,
                    "outcome": outcome,
                    "coef": coef,
                    "stderr": stderr,
                }
            )
            sig = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "n.s."
            report_lines.append(
                f"- {outcome}: coef={coef:+.6f}, p={pvalue:.4g}, R2={model.rsquared:.4f} {sig}"
            )

        report_lines.append("")
        results[family_name] = family_summary

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    plot_df = pd.DataFrame(plot_rows)
    plot_df["label"] = plot_df["family"] + "\n" + plot_df["outcome"]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        x=plot_df["coef"],
        y=range(len(plot_df)),
        xerr=1.96 * plot_df["stderr"],
        fmt="o",
        color="#7a1c1c",
        ecolor="#d69b9b",
        capsize=4,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Treatment effect (95% CI)")
    ax.set_title("P2b Tokenizer-Verified Matrix Effects")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close()

    print("✅ Saved P2b verified-matrix analysis.")


if __name__ == "__main__":
    main()
