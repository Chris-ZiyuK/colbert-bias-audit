#!/usr/bin/env python3
"""Auxiliary P2b validation using the names-dataset library.

This experiment is intentionally race-agnostic. It uses country + gender +
popularity-rank metadata from `names-dataset` to strengthen the language-side
explanations in P2b:
  1. tokenization effect under close popularity rank
  2. popularity-rank-gap effect under matched tokenizer count

It is not used to redefine race labels.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from names_dataset import NameDataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.audit.core import compute_tcd_breakdown, encode, get_token_count, get_tokens, load_model


OUT_DIR = ROOT / "results" / "p2_name_confound"
DATA_DIR = ROOT / "data" / "audit_names"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POOL_PATH = DATA_DIR / "p2b_aux_names_dataset_pool.json"
PAIR_PATH = DATA_DIR / "p2b_aux_names_dataset_pairs.json"
SUMMARY_PATH = OUT_DIR / "p2b_aux_names_dataset_summary.json"
REPORT_PATH = OUT_DIR / "p2b_aux_names_dataset_report.txt"
PLOT_PATH = OUT_DIR / "p2b_aux_names_dataset_effects.png"

COUNTRIES = ["US", "IN", "ES", "NG", "BR", "FR"]
GENDERS = [("Female", "F"), ("Male", "M")]
TOP_N = 15
PAIR_TARGET = 24
MAX_PER_GROUP = 3

TEMPLATES = [
    ("t1", "{name} has extensive experience in this field."),
    ("t2", "{name} is widely recognized as a leading professional."),
    ("t3", "{name} has been honored with multiple awards for outstanding work."),
]


def pair_id(a: str, b: str) -> str:
    left, right = sorted([a, b])
    return f"{left}-{right}"


def fit_clustered(data: pd.DataFrame, formula: str):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["pair_id"]},
    )


def select_balanced_pairs(candidates, sort_key, max_total=PAIR_TARGET, max_per_group=MAX_PER_GROUP):
    selected = []
    counts = {}
    seen_pairs = set()
    for item in sorted(candidates, key=sort_key):
        group = (item["country"], item["gender"])
        if counts.get(group, 0) >= max_per_group:
            continue
        if item["pair_id"] in seen_pairs:
            continue
        selected.append(item)
        counts[group] = counts.get(group, 0) + 1
        seen_pairs.add(item["pair_id"])
        if len(selected) >= max_total:
            break
    return selected


def build_pool():
    nd = NameDataset()
    tokenizer, model, device = load_model()

    pool = []
    for country in COUNTRIES:
        for gender_full, gender_short in GENDERS:
            top_names = nd.get_top_names(n=TOP_N, gender=gender_full, country_alpha2=country)[country][gender_full[0]]
            for rank, name in enumerate(top_names, start=1):
                pool.append(
                    {
                        "name": name,
                        "country": country,
                        "gender": gender_short,
                        "rank": rank,
                        "token_count": get_token_count(name, tokenizer),
                    }
                )
    return pool, tokenizer, model, device


def build_pairs(pool):
    token_treatment = []
    token_control = []
    rank_treatment = []
    rank_control = []

    grouped = {}
    for item in pool:
        grouped.setdefault((item["country"], item["gender"]), []).append(item)

    for (country, gender), rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["rank"])
        for a, b in combinations(rows, 2):
            rank_gap = abs(a["rank"] - b["rank"])
            token_diff = abs(a["token_count"] - b["token_count"])
            record = {
                "pair_id": pair_id(a["name"], b["name"]),
                "country": country,
                "gender": gender,
                "name_a": a["name"],
                "name_b": b["name"],
                "rank_a": a["rank"],
                "rank_b": b["rank"],
                "rank_gap": rank_gap,
                "token_a": a["token_count"],
                "token_b": b["token_count"],
                "token_diff": token_diff,
            }

            if rank_gap <= 5 and token_diff > 0:
                token_treatment.append(record)
            if rank_gap <= 5 and token_diff == 0:
                token_control.append(record)
            if token_diff == 0 and rank_gap >= 8:
                rank_treatment.append(record)
            if token_diff == 0 and rank_gap <= 2:
                rank_control.append(record)

    selected = {
        "tokenization_treatment": select_balanced_pairs(token_treatment, sort_key=lambda x: (x["rank_gap"], -x["token_diff"])),
        "tokenization_control": select_balanced_pairs(token_control, sort_key=lambda x: (x["rank_gap"], x["pair_id"])),
        "rankgap_treatment": select_balanced_pairs(rank_treatment, sort_key=lambda x: (-x["rank_gap"], x["pair_id"])),
        "rankgap_control": select_balanced_pairs(rank_control, sort_key=lambda x: (x["rank_gap"], x["pair_id"])),
    }
    return selected


def run_experiment(pool, pair_sets, tokenizer, model, device):
    with open(ROOT / "data" / "professions" / "professions.json") as handle:
        professions = json.load(handle)["professions"]

    profession_lookup = {p["name"]: p for p in professions}
    queries = {p["name"]: p["query"] for p in professions}

    unique_names = {row["name"] for row in pool}
    query_cache = {}
    for profession, query_text in queries.items():
        query_cache[profession] = {
            "emb": encode(query_text, tokenizer, model, device, is_query=True),
            "tokens": get_tokens(query_text, tokenizer, is_query=True),
        }

    doc_cache = {}
    for template_id, template in TEMPLATES:
        for name in unique_names:
            doc_text = template.format(name=name)
            doc_cache[(template_id, name)] = {
                "text": doc_text,
                "emb": encode(doc_text, tokenizer, model, device),
            }

    family_map = {
        "tokenization_treatment": ("tokenization", 1),
        "tokenization_control": ("tokenization", 0),
        "rankgap_treatment": ("rankgap", 1),
        "rankgap_control": ("rankgap", 0),
    }

    results = []
    for pair_group, items in pair_sets.items():
        family, treatment = family_map[pair_group]
        for item in items:
            for profession, query_text in queries.items():
                q_emb = query_cache[profession]["emb"]
                q_tokens = query_cache[profession]["tokens"]
                prof_meta = profession_lookup[profession]
                for template_id, _ in TEMPLATES:
                    d1 = doc_cache[(template_id, item["name_a"])]
                    d2 = doc_cache[(template_id, item["name_b"])]
                    breakdown = compute_tcd_breakdown(q_emb, d1["emb"], d2["emb"], q_tokens)
                    results.append(
                        {
                            "family": family,
                            "treatment": treatment,
                            "pair_group": pair_group,
                            "pair_id": item["pair_id"],
                            "country": item["country"],
                            "gender": item["gender"],
                            "name_a": item["name_a"],
                            "name_b": item["name_b"],
                            "rank_gap": item["rank_gap"],
                            "token_diff": item["token_diff"],
                            "profession": profession,
                            "template_id": template_id,
                            "bls_category": prof_meta["bls_category"],
                            "ss": breakdown["ss"],
                            "func_tcd": breakdown["func_tcd"],
                            "cont_tcd": breakdown["cont_tcd"],
                            "tcd_ratio": breakdown["tcd_ratio"],
                        }
                    )
    return pd.DataFrame(results)


def summarize(df):
    summary = {}
    plot_rows = []
    lines = [
        "P2b auxiliary names-dataset validation",
        "=" * 60,
        "",
    ]

    for family in sorted(df["family"].unique()):
        sub = df[df["family"] == family].copy()
        means = sub.groupby("treatment")[["ss", "func_tcd", "cont_tcd"]].mean().round(6)
        family_summary = {
            "n_rows": int(len(sub)),
            "n_pairs": int(sub["pair_id"].nunique()),
            "means": means.reset_index().to_dict(orient="records"),
            "models": {},
        }
        lines.append(family)
        lines.append(f"- rows={len(sub)}, pairs={sub['pair_id'].nunique()}")
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
                    "family": family,
                    "outcome": outcome,
                    "coef": coef,
                    "stderr": stderr,
                }
            )
        lines.append("")
        summary[family] = family_summary

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    plot_df = pd.DataFrame(plot_rows)
    plot_df["label"] = plot_df["family"] + "\n" + plot_df["outcome"]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        x=plot_df["coef"],
        y=range(len(plot_df)),
        xerr=1.96 * plot_df["stderr"],
        fmt="o",
        color="#0b6e99",
        ecolor="#8ab9ce",
        capsize=4,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Treatment effect (95% CI)")
    ax.set_title("P2b auxiliary names-dataset validation")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    pool, tokenizer, model, device = build_pool()
    pair_sets = build_pairs(pool)

    with POOL_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "_doc": "Auxiliary pool from names-dataset for race-agnostic P2b validation.",
                "source": "names-dataset 3.3.1",
                "countries": COUNTRIES,
                "top_n_per_country_gender": TOP_N,
                "names": pool,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with PAIR_PATH.open("w", encoding="utf-8") as f:
        json.dump(pair_sets, f, indent=2, ensure_ascii=False)

    df = run_experiment(pool, pair_sets, tokenizer, model, device)
    summarize(df)
    print("✅ Saved P2b auxiliary names-dataset results.")


if __name__ == "__main__":
    main()
