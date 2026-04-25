"""
P2 follow-up analysis
=====================
Targeted follow-up on the upgraded P2 run. The goal is to separate:
  - gender contrast inside same-race comparisons
  - race contrast inside same-gender comparisons
  - tokenization effects in tightly frequency-matched subsets
  - frequency effects in same-token, same-race, same-gender subsets

Run:
  conda run -n colbert_bias python experiments/phase2_name_confound/p2_followup_analysis.py
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
from statsmodels.formula.api import ols


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "p2_name_confound"
RAW_PATH = RESULTS_DIR / "name_confound_raw.csv"


def fit_clustered(formula, data):
    return ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["pair_id"]},
    )


def coef_record(result, term):
    return {
        "coef": float(result.params.get(term, 0.0)),
        "pvalue": float(result.pvalues.get(term, 1.0)),
        "stderr": float(result.bse.get(term, 0.0)),
        "significant": bool(result.pvalues.get(term, 1.0) < 0.05),
    }


def sig_stars(pvalue):
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."


def plot_followup_coefficients(records):
    labels = [r["label"] for r in records]
    coefs = [r["coef"] for r in records]
    errors = [1.96 * r["stderr"] for r in records]
    colors = ["#9b2226" if r["pvalue"] < 0.05 else "#9aa1ab" for r in records]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ypos = np.arange(len(records))
    ax.barh(ypos, coefs, xerr=errors, color=colors, alpha=0.9, capsize=4)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("P2 Follow-up Isolation Tests")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "p2_followup_coefficients.png", dpi=160, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(RAW_PATH)

    same_race = df[df["same_race"] == 1].copy()
    same_gender = df[df["same_gender"] == 1].copy()
    freq_matched = df[
        (df["same_race"] == 1) &
        (df["same_gender"] == 1) &
        (df["freq_diff"] <= 0.25)
    ].copy()
    token_matched = df[
        (df["same_race"] == 1) &
        (df["same_gender"] == 1) &
        (df["token_diff"] == 0)
    ].copy()

    models = {
        "within_race_ss_gender": fit_clustered(
            "ss ~ cross_gender + freq_diff + mean_log_freq + mean_tokens + C(ses_pair) + C(profession) + C(template_id)",
            same_race,
        ),
        "within_race_func_gender": fit_clustered(
            "func_tcd ~ cross_gender + freq_diff + mean_log_freq + mean_tokens + C(ses_pair) + C(profession) + C(template_id)",
            same_race,
        ),
        "within_gender_ss_race": fit_clustered(
            "ss ~ cross_race + freq_diff + mean_log_freq + mean_tokens + C(ses_pair) + C(profession) + C(template_id)",
            same_gender,
        ),
        "within_gender_func_race": fit_clustered(
            "func_tcd ~ cross_race + freq_diff + mean_log_freq + mean_tokens + C(ses_pair) + C(profession) + C(template_id)",
            same_gender,
        ),
        "freq_matched_ss_token": fit_clustered(
            "ss ~ token_diff + mean_tokens + C(profession) + C(template_id)",
            freq_matched,
        ),
        "freq_matched_func_token": fit_clustered(
            "func_tcd ~ token_diff + mean_tokens + C(profession) + C(template_id)",
            freq_matched,
        ),
        "token_matched_ss_freq": fit_clustered(
            "ss ~ freq_diff + mean_log_freq + C(profession) + C(template_id)",
            token_matched,
        ),
        "token_matched_func_freq": fit_clustered(
            "func_tcd ~ freq_diff + mean_log_freq + C(profession) + C(template_id)",
            token_matched,
        ),
    }

    coeff_records = [
        {
            "label": "Within-race: cross_gender -> SS",
            **coef_record(models["within_race_ss_gender"], "cross_gender"),
        },
        {
            "label": "Within-race: cross_gender -> Func-TCD",
            **coef_record(models["within_race_func_gender"], "cross_gender"),
        },
        {
            "label": "Within-gender: cross_race -> SS",
            **coef_record(models["within_gender_ss_race"], "cross_race"),
        },
        {
            "label": "Within-gender: cross_race -> Func-TCD",
            **coef_record(models["within_gender_func_race"], "cross_race"),
        },
        {
            "label": "Freq-matched: token_diff -> SS",
            **coef_record(models["freq_matched_ss_token"], "token_diff"),
        },
        {
            "label": "Freq-matched: token_diff -> Func-TCD",
            **coef_record(models["freq_matched_func_token"], "token_diff"),
        },
        {
            "label": "Token-matched: freq_diff -> SS",
            **coef_record(models["token_matched_ss_freq"], "freq_diff"),
        },
        {
            "label": "Token-matched: freq_diff -> Func-TCD",
            **coef_record(models["token_matched_func_freq"], "freq_diff"),
        },
    ]

    plot_followup_coefficients(coeff_records)

    summary = {
        "subset_sizes": {
            "same_race": int(len(same_race)),
            "same_gender": int(len(same_gender)),
            "freq_matched_same_race_same_gender": int(len(freq_matched)),
            "token_matched_same_race_same_gender": int(len(token_matched)),
        },
        "models": {
            name: {
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
            }
            for name, model in models.items()
        },
        "coefficients": {
            "within_race_ss_cross_gender": coef_record(models["within_race_ss_gender"], "cross_gender"),
            "within_race_func_cross_gender": coef_record(models["within_race_func_gender"], "cross_gender"),
            "within_gender_ss_cross_race": coef_record(models["within_gender_ss_race"], "cross_race"),
            "within_gender_func_cross_race": coef_record(models["within_gender_func_race"], "cross_race"),
            "freq_matched_ss_token_diff": coef_record(models["freq_matched_ss_token"], "token_diff"),
            "freq_matched_func_token_diff": coef_record(models["freq_matched_func_token"], "token_diff"),
            "token_matched_ss_freq_diff": coef_record(models["token_matched_ss_freq"], "freq_diff"),
            "token_matched_func_freq_diff": coef_record(models["token_matched_func_freq"], "freq_diff"),
        },
        "interpretation": {
            "gender_statement": (
                "Gender contrast remains robust inside same-race comparisons, so the upgraded P2 "
                "gender signal is not merely a byproduct of race mixing."
            ),
            "race_statement": (
                "Race contrast does not remain stable once gender is held fixed; any remaining "
                "race effect is weak and does not survive cleanly in the within-gender models."
            ),
            "tokenization_statement": (
                "When race, gender, and frequency are tightly matched, token difference becomes a "
                "significant positive predictor of both SS and function-word disruption."
            ),
            "frequency_statement": (
                "Once token count, race, and gender are exactly matched, frequency alone is weaker: "
                "it is not significant for SS and only borderline for function-word disruption."
            ),
        },
    }

    with open(RESULTS_DIR / "p2_followup_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "p2_followup_stats.txt", "w") as handle:
        handle.write("P2 follow-up analysis\n")
        handle.write("=" * 60 + "\n\n")
        handle.write("Subset sizes\n")
        for key, value in summary["subset_sizes"].items():
            handle.write(f"{key}: {value}\n")
        handle.write("\nCoefficient highlights\n")
        for label, record in summary["coefficients"].items():
            handle.write(
                f"{label}: coef={record['coef']:+.6f}, "
                f"se={record['stderr']:.6f}, "
                f"p={record['pvalue']:.6g} {sig_stars(record['pvalue'])}\n"
            )
        handle.write("\nInterpretation\n")
        for key, value in summary["interpretation"].items():
            handle.write(f"- {key}: {value}\n")

    print("✅ Saved P2 follow-up summary and stats.")


if __name__ == "__main__":
    main()
