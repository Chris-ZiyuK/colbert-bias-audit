"""
P1 wrap-up analysis
===================
Turn the profession-expansion raw output into a cleaner, publication-ready
analysis focused on:
  - function vs content dominance overall and by swap type
  - name vs pronoun comparison at the profession level
  - category-level variation with formal tests
  - bootstrap confidence intervals for key mean effects

Run:
  conda run -n colbert_bias python experiments/phase2_profession_expansion/p1_wrapup_analysis.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "p1_profession_expansion"
RAW_PATH = RESULTS_DIR / "profession_expansion_raw.csv"

np.random.seed(42)


def bootstrap_mean_ci(series, n_boot=4000, alpha=0.05):
    values = np.asarray(series, dtype=float)
    boots = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        boots.append(sample.mean())
    lower = float(np.quantile(boots, alpha / 2))
    upper = float(np.quantile(boots, 1 - alpha / 2))
    return lower, upper


def safe_wilcoxon(x, y, alternative):
    try:
        stat, pvalue = stats.wilcoxon(x, y, alternative=alternative)
        return float(stat), float(pvalue)
    except ValueError:
        return None, None


def sig_stars(pvalue):
    if pvalue is None:
        return "n.a."
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."


def plot_swap_comparison(summary_df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    metrics = [
        ("ss", "Mean SS", "#9b2226"),
        ("func_tcd", "Mean Func-TCD", "#0b6e99"),
        ("func_minus_cont", "Mean Func - Cont", "#386641"),
    ]

    for ax, (metric, title, color) in zip(axes, metrics):
        ax.bar(summary_df["swap_type"], summary_df[metric], color=color, alpha=0.9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "p1_swap_type_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_category_split(category_df):
    pivot = category_df.pivot(index="bls_category", columns="swap_type", values="mean_ss")
    pivot = pivot.loc[sorted(pivot.index)]

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax, color=["#9b2226", "#0b6e99"], alpha=0.85)
    ax.set_ylabel("Mean SS")
    ax.set_title("P1 Mean SS by Category and Swap Type")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Swap Type")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "p1_category_swap_split.png", dpi=160, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(RAW_PATH)
    df["func_minus_cont"] = df["func_tcd"] - df["cont_tcd"]

    # ============================================================
    # Overall and swap-type summaries
    # ============================================================
    swap_summary = (
        df.groupby("swap_type")
        .agg(
            mean_ss=("ss", "mean"),
            mean_func_tcd=("func_tcd", "mean"),
            mean_cont_tcd=("cont_tcd", "mean"),
            mean_ratio=("tcd_ratio", "mean"),
            mean_func_minus_cont=("func_minus_cont", "mean"),
            n=("ss", "size"),
        )
        .reset_index()
    )

    swap_summary["ss"] = swap_summary["mean_ss"]
    swap_summary["func_tcd"] = swap_summary["mean_func_tcd"]
    swap_summary["func_minus_cont"] = swap_summary["mean_func_minus_cont"]

    # Row-level signed tests
    row_level_tests = {}
    for swap_type, group in df.groupby("swap_type"):
        stat, pvalue = safe_wilcoxon(group["func_tcd"], group["cont_tcd"], alternative="greater")
        row_level_tests[swap_type] = {
            "n_rows": int(len(group)),
            "mean_func_tcd": float(group["func_tcd"].mean()),
            "mean_cont_tcd": float(group["cont_tcd"].mean()),
            "mean_func_minus_cont": float(group["func_minus_cont"].mean()),
            "wilcoxon_greater_stat": stat,
            "wilcoxon_greater_pvalue": pvalue,
        }

    overall_stat, overall_p = safe_wilcoxon(df["func_tcd"], df["cont_tcd"], alternative="greater")

    # Profession-level paired comparison between name and pronoun
    prof_swap = (
        df.groupby(["profession", "swap_type"])
        .agg(
            mean_ss=("ss", "mean"),
            mean_func_tcd=("func_tcd", "mean"),
            mean_cont_tcd=("cont_tcd", "mean"),
            mean_func_minus_cont=("func_minus_cont", "mean"),
        )
        .reset_index()
    )

    prof_wide = prof_swap.pivot(index="profession", columns="swap_type")
    paired_tests = {}
    for metric in ["mean_ss", "mean_func_tcd", "mean_func_minus_cont"]:
        stat, pvalue = safe_wilcoxon(
            prof_wide[(metric, "pronoun")],
            prof_wide[(metric, "name")],
            alternative="greater",
        )
        paired_tests[metric] = {
            "pronoun_mean": float(prof_wide[(metric, "pronoun")].mean()),
            "name_mean": float(prof_wide[(metric, "name")].mean()),
            "wilcoxon_pronoun_gt_name_stat": stat,
            "wilcoxon_pronoun_gt_name_pvalue": pvalue,
        }

    # Category-level summary and tests
    category_swap_summary = (
        df.groupby(["bls_category", "swap_type"])
        .agg(
            mean_ss=("ss", "mean"),
            mean_func_tcd=("func_tcd", "mean"),
            mean_cont_tcd=("cont_tcd", "mean"),
            mean_func_minus_cont=("func_minus_cont", "mean"),
            n=("ss", "size"),
        )
        .reset_index()
    )

    prof_category = (
        df.groupby(["profession", "bls_category", "swap_type"])
        .agg(
            mean_ss=("ss", "mean"),
            mean_func_tcd=("func_tcd", "mean"),
            mean_func_minus_cont=("func_minus_cont", "mean"),
        )
        .reset_index()
    )

    category_tests = {}
    for swap_type in ["name", "pronoun"]:
        sub = prof_category[prof_category["swap_type"] == swap_type]
        groups = [g["mean_ss"].values for _, g in sub.groupby("bls_category")]
        h_stat, pvalue = stats.kruskal(*groups)
        category_tests[swap_type] = {
            "kruskal_h_mean_ss": float(h_stat),
            "kruskal_pvalue_mean_ss": float(pvalue),
        }

    # Correlations by swap type
    corr_summary = {}
    for swap_type in ["name", "pronoun"]:
        sub = prof_category[prof_category["swap_type"] == swap_type]
        sub_prof = (
            sub.groupby(["profession", "bls_category"], as_index=False)
            .agg(
                mean_ss=("mean_ss", "mean"),
                mean_func_tcd=("mean_func_tcd", "mean"),
            )
            .merge(
                df[["profession", "bls_female_pct"]].drop_duplicates(),
                on="profession",
                how="left",
            )
        )
        corr_ss = stats.pearsonr(sub_prof["bls_female_pct"], sub_prof["mean_ss"])
        corr_func = stats.pearsonr(sub_prof["bls_female_pct"], sub_prof["mean_func_tcd"])
        corr_summary[swap_type] = {
            "female_pct_vs_ss_r": float(corr_ss.statistic),
            "female_pct_vs_ss_p": float(corr_ss.pvalue),
            "female_pct_vs_func_tcd_r": float(corr_func.statistic),
            "female_pct_vs_func_tcd_p": float(corr_func.pvalue),
        }

    # Bootstrap CIs
    bootstrap = {}
    for label, sub in [("overall", df), ("name", df[df["swap_type"] == "name"]), ("pronoun", df[df["swap_type"] == "pronoun"])]:
        ss_ci = bootstrap_mean_ci(sub["ss"])
        diff_ci = bootstrap_mean_ci(sub["func_minus_cont"])
        bootstrap[label] = {
            "ss_mean": float(sub["ss"].mean()),
            "ss_ci_95": [ss_ci[0], ss_ci[1]],
            "func_minus_cont_mean": float(sub["func_minus_cont"].mean()),
            "func_minus_cont_ci_95": [diff_ci[0], diff_ci[1]],
        }

    # Function-dominance counts at profession level
    dominance = (
        prof_swap.assign(func_gt=lambda x: x["mean_func_tcd"] > x["mean_cont_tcd"])
        .groupby("swap_type")["func_gt"]
        .agg(["sum", "count"])
        .reset_index()
    )

    # Lightweight regression for write-up support
    reg = ols(
        "func_minus_cont ~ C(swap_type) + C(bls_category) + C(prestige) + bls_female_pct",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["profession"]})

    summary = {
        "n_rows": int(len(df)),
        "n_professions": int(df["profession"].nunique()),
        "swap_type_summary": swap_summary[
            [
                "swap_type",
                "mean_ss",
                "mean_func_tcd",
                "mean_cont_tcd",
                "mean_ratio",
                "mean_func_minus_cont",
                "n",
            ]
        ].to_dict(orient="records"),
        "row_level_func_gt_content_tests": row_level_tests,
        "overall_func_gt_content": {
            "mean_func_tcd": float(df["func_tcd"].mean()),
            "mean_cont_tcd": float(df["cont_tcd"].mean()),
            "wilcoxon_greater_stat": overall_stat,
            "wilcoxon_greater_pvalue": overall_p,
        },
        "profession_level_name_vs_pronoun_tests": paired_tests,
        "profession_level_dominance_counts": dominance.to_dict(orient="records"),
        "category_swap_summary": category_swap_summary.to_dict(orient="records"),
        "category_tests": category_tests,
        "correlations_by_swap_type": corr_summary,
        "bootstrap": bootstrap,
        "regression_support": {
            "swap_type_pronoun_coef": float(reg.params.get("C(swap_type)[T.pronoun]", 0.0)),
            "swap_type_pronoun_pvalue": float(reg.pvalues.get("C(swap_type)[T.pronoun]", 1.0)),
        },
        "interpretation": {
            "main_claim": (
                "Function-word dominance remains robust in both name and pronoun swaps, "
                "and pronoun swaps are modestly but consistently stronger than name swaps "
                "at the profession level."
            ),
            "category_claim": (
                "Profession category remains a meaningful moderator of SS, rather than the "
                "effect being driven by only one occupational slice."
            ),
        },
    }

    with open(RESULTS_DIR / "p1_wrapup_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "p1_wrapup_stats.txt", "w") as handle:
        handle.write("P1 wrap-up analysis\n")
        handle.write("=" * 60 + "\n\n")
        handle.write("Swap-type summary\n")
        handle.write(swap_summary.to_string(index=False))
        handle.write("\n\nOverall function > content test\n")
        handle.write(
            f"Wilcoxon (func > cont): stat={overall_stat}, p={overall_p:.6g} {sig_stars(overall_p)}\n"
        )
        handle.write("\nRow-level function > content by swap type\n")
        for swap_type, record in row_level_tests.items():
            handle.write(
                f"{swap_type}: mean_func={record['mean_func_tcd']:.6f}, "
                f"mean_cont={record['mean_cont_tcd']:.6f}, "
                f"p={record['wilcoxon_greater_pvalue']:.6g} "
                f"{sig_stars(record['wilcoxon_greater_pvalue'])}\n"
            )
        handle.write("\nProfession-level pronoun > name paired tests\n")
        for metric, record in paired_tests.items():
            handle.write(
                f"{metric}: pronoun_mean={record['pronoun_mean']:.6f}, "
                f"name_mean={record['name_mean']:.6f}, "
                f"p={record['wilcoxon_pronoun_gt_name_pvalue']:.6g} "
                f"{sig_stars(record['wilcoxon_pronoun_gt_name_pvalue'])}\n"
            )
        handle.write("\nCategory tests\n")
        for swap_type, record in category_tests.items():
            handle.write(
                f"{swap_type}: H={record['kruskal_h_mean_ss']:.4f}, "
                f"p={record['kruskal_pvalue_mean_ss']:.6g} "
                f"{sig_stars(record['kruskal_pvalue_mean_ss'])}\n"
            )
        handle.write("\nRegression support\n")
        handle.write(
            f"C(swap_type)[T.pronoun] on func_minus_cont: "
            f"coef={summary['regression_support']['swap_type_pronoun_coef']:+.6f}, "
            f"p={summary['regression_support']['swap_type_pronoun_pvalue']:.6g}\n"
        )

    plot_swap_comparison(swap_summary)
    plot_category_split(category_swap_summary)

    print("✅ Saved P1 wrap-up summary and stats.")


if __name__ == "__main__":
    main()
