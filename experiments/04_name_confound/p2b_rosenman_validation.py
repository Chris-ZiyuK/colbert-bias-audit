#!/usr/bin/env python3
"""Run targeted P2b validation on the Rosenman-backed expansion matrix."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.audit.core import compute_tcd_breakdown, encode, get_tokens, load_model


MATRIX_PATH = ROOT / "data" / "audit_names" / "p2b_rosenman_matrix.json"
OUT_DIR = ROOT / "results" / "p2_name_confound"
ROWS_PATH = OUT_DIR / "p2b_rosenman_rows.csv"
SUMMARY_PATH = OUT_DIR / "p2b_rosenman_summary.json"
REPORT_PATH = OUT_DIR / "p2b_rosenman_report.txt"
PLOT_PATH = OUT_DIR / "p2b_rosenman_effects.png"

TEMPLATES = [
    ("t1", "{name} has extensive experience in this field."),
    ("t2", "{name} is widely recognized as a leading professional."),
    ("t3", "{name} has been honored with multiple awards for outstanding work."),
]


def fit_clustered(data: pd.DataFrame, formula: str):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["pair_id"]},
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MATRIX_PATH.open() as handle:
        families = json.load(handle)["families"]

    with open(ROOT / "data" / "professions" / "professions.json") as handle:
        professions = json.load(handle)["professions"]

    profession_lookup = {p["name"]: p for p in professions}
    queries = {p["name"]: p["query"] for p in professions}

    tokenizer, model, device = load_model()

    unique_names = set()
    for family in families.values():
        for pair in family["pairs"]:
            unique_names.add(pair["name_a"])
            unique_names.add(pair["name_b"])

    query_cache = {}
    for profession, query_text in queries.items():
        query_cache[profession] = {
            "emb": encode(query_text, tokenizer, model, device, is_query=True),
            "tokens": get_tokens(query_text, tokenizer, is_query=True),
        }

    doc_cache = {}
    for template_id, template in TEMPLATES:
        for name in unique_names:
            text = template.format(name=name)
            doc_cache[(template_id, name)] = {
                "text": text,
                "emb": encode(text, tokenizer, model, device),
            }

    rows = []
    for family_name, family in families.items():
        for item in family["pairs"]:
            for profession, query_text in queries.items():
                q_emb = query_cache[profession]["emb"]
                q_tokens = query_cache[profession]["tokens"]
                prof_meta = profession_lookup[profession]
                for template_id, _ in TEMPLATES:
                    d1 = doc_cache[(template_id, item["name_a"])]
                    d2 = doc_cache[(template_id, item["name_b"])]
                    breakdown = compute_tcd_breakdown(q_emb, d1["emb"], d2["emb"], q_tokens)
                    rows.append(
                        {
                            "family": family_name,
                            "pair_id": item["pair_id"],
                            "treatment": item["treatment"],
                            "group": item["group"],
                            "name_a": item["name_a"],
                            "name_b": item["name_b"],
                            "race_a": item["race_a"],
                            "race_b": item["race_b"],
                            "gender_a": item["gender_a"],
                            "gender_b": item["gender_b"],
                            "tokens_a": item["tokens_a"],
                            "tokens_b": item["tokens_b"],
                            "freq_diff": item["freq_diff"],
                            "mean_log_freq": item["mean_log_freq"],
                            "profession": profession,
                            "template_id": template_id,
                            "bls_category": prof_meta["bls_category"],
                            "ss": breakdown["ss"],
                            "func_tcd": breakdown["func_tcd"],
                            "cont_tcd": breakdown["cont_tcd"],
                            "tcd_ratio": breakdown["tcd_ratio"],
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(ROWS_PATH, index=False)

    summary = {}
    plot_rows = []
    lines = [
        "P2b Rosenman targeted validation",
        "=" * 60,
        "",
    ]

    for family_name in ("race", "gender", "tokenization", "frequency_gap", "absolute_rarity"):
        sub = df[df["family"] == family_name].copy()
        if sub.empty:
            continue

        pair_count = int(sub["pair_id"].nunique())
        family_summary = {
            "n_rows": int(len(sub)),
            "n_pairs": pair_count,
            "mean_by_group": (
                sub.groupby("group")[["ss", "func_tcd", "cont_tcd"]]
                .mean()
                .round(6)
                .reset_index()
                .to_dict(orient="records")
            ),
            "models": {},
        }

        lines.append(family_name)
        lines.append(f"- rows={len(sub)}, pairs={pair_count}")
        for outcome in ("ss", "func_tcd"):
            model = fit_clustered(sub, f"{outcome} ~ treatment + C(profession) + C(template_id)")
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
            lines.append(f"- {outcome}: coef={coef:+.6f}, p={pvalue:.4g}, R2={model.rsquared:.4f}")
            plot_rows.append(
                {
                    "family": family_name,
                    "outcome": outcome,
                    "coef": coef,
                    "stderr": stderr,
                }
            )
        lines.append("")
        summary[family_name] = family_summary

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    plot_df = pd.DataFrame(plot_rows)
    plot_df["label"] = plot_df["family"] + "\n" + plot_df["outcome"]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(
        x=plot_df["coef"],
        y=range(len(plot_df)),
        xerr=1.96 * plot_df["stderr"],
        fmt="o",
        color="#145a6b",
        ecolor="#8fc0ca",
        capsize=4,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Treatment effect (95% CI)")
    ax.set_title("P2b Rosenman targeted validation")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close()

    print("✅ Saved P2b Rosenman validation outputs.")


if __name__ == "__main__":
    main()
