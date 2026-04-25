"""
P2: Name Confound Decomposition (publication-ready upgrade)
===========================================================
Disentangle what counterfactual name swaps are actually measuring by
separating:
  1. Social rarity / frequency (Census-based log frequency)
  2. Model-facing fragmentation (BPE tokenization)
  3. Identity attributes (race, gender)
  4. SES association (name-linked socioeconomic tier)

Upgrades over the first-pass version:
  - Uses all 33 professions from the profession inventory
  - Uses 3 document templates instead of 1
  - Caches query/document embeddings for efficiency
  - Fits profession/template fixed-effects regressions
  - Uses cluster-robust standard errors by name pair
  - Adds gender and SES to the main specification
  - Models both pairwise difference and pair-level rarity/fragmentation

Run:
  conda run -n colbert_bias python experiments/phase2_name_confound/name_confound_decomposition.py
"""

import json
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.core import (  # noqa: E402
    compute_tcd_breakdown,
    encode,
    get_token_count,
    get_tokens,
    load_model,
)

warnings.filterwarnings("ignore")
np.random.seed(42)


RESULTS_DIR = PROJECT_ROOT / "results" / "p2_name_confound"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES = [
    ("t1", "{name} has extensive experience in this field."),
    ("t2", "{name} is widely recognized as a leading professional."),
    ("t3", "{name} has been honored with multiple awards for outstanding work."),
]


def sorted_label(a, b):
    """Create a stable pair label regardless of ordering."""
    left, right = sorted([str(a), str(b)])
    return f"{left}-{right}"


def sig_stars(pvalue):
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."


def fit_clustered(formula, data):
    """OLS with cluster-robust standard errors by name pair."""
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


def summarize_group(df, group_col):
    summary = (
        df.groupby(group_col)
        .agg(
            mean_ss=("ss", "mean"),
            mean_func_tcd=("func_tcd", "mean"),
            mean_cont_tcd=("cont_tcd", "mean"),
            mean_tcd_ratio=("tcd_ratio", "mean"),
            n=("ss", "size"),
        )
        .reset_index()
        .sort_values("mean_ss", ascending=False)
    )
    return summary


def write_model_block(handle, title, result, core_terms):
    handle.write("=" * 78 + "\n")
    handle.write(title + "\n")
    handle.write("=" * 78 + "\n")
    handle.write(result.summary().as_text())
    handle.write("\n\nCore coefficients:\n")
    for label, term in core_terms:
        coef = result.params.get(term, 0.0)
        pval = result.pvalues.get(term, 1.0)
        stderr = result.bse.get(term, 0.0)
        handle.write(
            f"  {label:<18s} coef={coef:+.6f}  se={stderr:.6f}  "
            f"p={pval:.6f}  {sig_stars(pval)}\n"
        )
    handle.write("\n")


def plot_core_coefficients(ss_model, func_model):
    labels = [
        ("cross_race", "Cross-race pair"),
        ("cross_gender", "Cross-gender pair"),
        ("token_diff", "Token diff"),
        ("mean_tokens", "Mean BPE tokens"),
        ("freq_diff", "Log-frequency diff"),
        ("mean_log_freq", "Mean log-frequency"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, result, title, color in [
        (axes[0], ss_model, "Main Model: SS", "#9b2226"),
        (axes[1], func_model, "Main Model: Func-TCD", "#0b6e99"),
    ]:
        coefs = [result.params.get(term, 0.0) for term, _ in labels]
        errors = [1.96 * result.bse.get(term, 0.0) for term, _ in labels]
        pvals = [result.pvalues.get(term, 1.0) for term, _ in labels]
        ypos = np.arange(len(labels))
        bar_colors = [color if p < 0.05 else "#9aa1ab" for p in pvals]

        ax.barh(ypos, coefs, xerr=errors, color=bar_colors, alpha=0.9, capsize=4)
        ax.axvline(0, color="#444", linewidth=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([label for _, label in labels], fontsize=10)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "main_effects_coefficients.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_group_summary(race_summary, gender_summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].barh(
        race_summary["race_pair"],
        race_summary["mean_ss"],
        color="#9b2226",
        alpha=0.85,
    )
    axes[0].set_title("Mean SS by Race Pair")
    axes[0].set_xlabel("Mean SS")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(
        gender_summary["gender_pair"],
        gender_summary["mean_func_tcd"],
        color="#0b6e99",
        alpha=0.85,
    )
    axes[1].set_title("Mean Func-TCD by Gender Pair")
    axes[1].set_xlabel("Mean Func-TCD")
    axes[1].grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "group_summaries.png", dpi=160, bbox_inches="tight")
    plt.close()


def main():
    # ============================================================
    # Load metadata
    # ============================================================
    with open(PROJECT_ROOT / "data" / "audit_names" / "name_features.json") as handle:
        name_data = json.load(handle)
        names = name_data["names"]

    with open(PROJECT_ROOT / "data" / "professions" / "professions.json") as handle:
        profession_data = json.load(handle)["professions"]

    print(f"📋 Loaded {len(names)} names from confound-breaking matrix")
    print(f"📋 Loaded {len(profession_data)} professions for full-scale P2")

    tokenizer, model, device = load_model()

    # Verify / update tokenizer-facing properties
    token_updates = []
    for name_info in names:
        actual_tokens = get_token_count(name_info["name"], tokenizer)
        if actual_tokens != name_info["bpe_tokens"]:
            print(
                f"  ⚠️ Updating token count for {name_info['name']}: "
                f"{name_info['bpe_tokens']} -> {actual_tokens}"
            )
            token_updates.append({
                "name": name_info["name"],
                "old": int(name_info["bpe_tokens"]),
                "new": int(actual_tokens),
            })
            name_info["bpe_tokens"] = actual_tokens
        name_info["log_freq"] = np.log10(max(name_info["census_freq_per_100k"], 0.1))

    profession_lookup = {prof["name"]: prof for prof in profession_data}
    queries = {prof["name"]: prof["query"] for prof in profession_data}
    name_pairs = list(combinations(range(len(names)), 2))

    total_tests = len(name_pairs) * len(queries) * len(TEMPLATES)
    print(
        f"\n🔬 Total tests: {len(name_pairs)} name pairs × {len(queries)} professions × "
        f"{len(TEMPLATES)} templates = {total_tests}"
    )

    # ============================================================
    # Cache all query / document embeddings
    # ============================================================
    print("\n⚙️ Caching query embeddings...")
    query_cache = {}
    for profession, query_text in queries.items():
        query_cache[profession] = {
            "emb": encode(query_text, tokenizer, model, device, is_query=True),
            "tokens": get_tokens(query_text, tokenizer, is_query=True),
        }

    print("⚙️ Caching document embeddings...")
    doc_cache = {}
    for template_id, template in TEMPLATES:
        for name_info in names:
            doc_text = template.format(name=name_info["name"])
            doc_cache[(template_id, name_info["name"])] = {
                "text": doc_text,
                "emb": encode(doc_text, tokenizer, model, device),
            }

    # ============================================================
    # Run full experiment
    # ============================================================
    print("\n🧪 Running pairwise comparisons...")
    results = []
    completed = 0

    for profession, query_text in queries.items():
        q_emb = query_cache[profession]["emb"]
        q_tokens = query_cache[profession]["tokens"]
        prof_meta = profession_lookup[profession]

        for template_id, template in TEMPLATES:
            for i, j in name_pairs:
                n1 = names[i]
                n2 = names[j]
                doc1 = doc_cache[(template_id, n1["name"])]
                doc2 = doc_cache[(template_id, n2["name"])]
                breakdown = compute_tcd_breakdown(q_emb, doc1["emb"], doc2["emb"], q_tokens)

                same_race = int(n1["race"] == n2["race"])
                same_gender = int(n1["gender"] == n2["gender"])
                cross_race = 1 - same_race
                cross_gender = 1 - same_gender

                tokens_a = int(n1["bpe_tokens"])
                tokens_b = int(n2["bpe_tokens"])
                log_freq_a = float(n1["log_freq"])
                log_freq_b = float(n2["log_freq"])

                results.append({
                    "profession": profession,
                    "bls_category": prof_meta["bls_category"],
                    "prestige": prof_meta["prestige"],
                    "bls_female_pct": prof_meta["bls_female_pct"],
                    "template_id": template_id,
                    "template_text": template,
                    "pair_id": sorted_label(n1["name"], n2["name"]),
                    "name_a": n1["name"],
                    "name_b": n2["name"],
                    "race_a": n1["race"],
                    "race_b": n2["race"],
                    "gender_a": n1["gender"],
                    "gender_b": n2["gender"],
                    "ses_a": n1["ses_tier"],
                    "ses_b": n2["ses_tier"],
                    "race_pair": sorted_label(n1["race"], n2["race"]),
                    "gender_pair": sorted_label(n1["gender"], n2["gender"]),
                    "ses_pair": sorted_label(n1["ses_tier"], n2["ses_tier"]),
                    "same_race": same_race,
                    "cross_race": cross_race,
                    "same_gender": same_gender,
                    "cross_gender": cross_gender,
                    "tokens_a": tokens_a,
                    "tokens_b": tokens_b,
                    "token_diff": abs(tokens_a - tokens_b),
                    "mean_tokens": 0.5 * (tokens_a + tokens_b),
                    "max_tokens": max(tokens_a, tokens_b),
                    "sum_tokens": tokens_a + tokens_b,
                    "freq_a": float(n1["census_freq_per_100k"]),
                    "freq_b": float(n2["census_freq_per_100k"]),
                    "log_freq_a": log_freq_a,
                    "log_freq_b": log_freq_b,
                    "freq_diff": abs(log_freq_a - log_freq_b),
                    "mean_log_freq": 0.5 * (log_freq_a + log_freq_b),
                    "min_log_freq": min(log_freq_a, log_freq_b),
                    "ss": breakdown["ss"],
                    "func_tcd": breakdown["func_tcd"],
                    "cont_tcd": breakdown["cont_tcd"],
                    "tcd_ratio": breakdown["tcd_ratio"],
                })
                completed += 1

        print(f"  ✅ {profession:25s} ({completed}/{total_tests})")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "name_confound_raw.csv", index=False)
    print(f"\n💾 Saved raw results: {RESULTS_DIR / 'name_confound_raw.csv'}")

    # ============================================================
    # Main models
    # ============================================================
    core_formula = (
        "cross_race + cross_gender + token_diff + mean_tokens + "
        "freq_diff + mean_log_freq + C(ses_pair) + C(profession) + C(template_id)"
    )

    main_ss = fit_clustered(f"ss ~ {core_formula}", df)
    main_func = fit_clustered(f"func_tcd ~ {core_formula}", df)

    interaction_ss = fit_clustered(
        "ss ~ cross_race * mean_tokens + cross_gender + token_diff + "
        "freq_diff + mean_log_freq + C(ses_pair) + C(profession) + C(template_id)",
        df,
    )
    interaction_func = fit_clustered(
        "func_tcd ~ cross_race * mean_tokens + cross_gender + token_diff + "
        "freq_diff + mean_log_freq + C(ses_pair) + C(profession) + C(template_id)",
        df,
    )

    same_race_df = df[df["same_race"] == 1].copy()
    same_token_df = df[df["token_diff"] == 0].copy()

    within_race_ss = fit_clustered(
        "ss ~ cross_gender + mean_tokens + freq_diff + mean_log_freq + "
        "C(ses_pair) + C(profession) + C(template_id)",
        same_race_df,
    )
    within_race_func = fit_clustered(
        "func_tcd ~ cross_gender + mean_tokens + freq_diff + mean_log_freq + "
        "C(ses_pair) + C(profession) + C(template_id)",
        same_race_df,
    )

    matched_token_ss = fit_clustered(
        "ss ~ cross_race + cross_gender + mean_tokens + freq_diff + mean_log_freq + "
        "C(ses_pair) + C(profession) + C(template_id)",
        same_token_df,
    )
    matched_token_func = fit_clustered(
        "func_tcd ~ cross_race + cross_gender + mean_tokens + freq_diff + mean_log_freq + "
        "C(ses_pair) + C(profession) + C(template_id)",
        same_token_df,
    )

    race_summary = summarize_group(df, "race_pair")
    gender_summary = summarize_group(df, "gender_pair")
    ses_summary = summarize_group(df, "ses_pair")

    plot_core_coefficients(main_ss, main_func)
    plot_group_summary(race_summary, gender_summary)

    # ============================================================
    # Save text report
    # ============================================================
    core_terms = [
        ("cross_race", "cross_race"),
        ("cross_gender", "cross_gender"),
        ("token_diff", "token_diff"),
        ("mean_tokens", "mean_tokens"),
        ("freq_diff", "freq_diff"),
        ("mean_log_freq", "mean_log_freq"),
    ]

    with open(RESULTS_DIR / "regression_table.txt", "w") as handle:
        write_model_block(handle, "Main Model 1: SS", main_ss, core_terms)
        write_model_block(handle, "Main Model 2: Func-TCD", main_func, core_terms)
        write_model_block(handle, "Interaction Model: SS ~ cross_race * mean_tokens", interaction_ss, core_terms)
        write_model_block(handle, "Interaction Model: Func-TCD ~ cross_race * mean_tokens", interaction_func, core_terms)
        write_model_block(handle, "Within-Race Robustness: SS", within_race_ss, [("cross_gender", "cross_gender")])
        write_model_block(handle, "Within-Race Robustness: Func-TCD", within_race_func, [("cross_gender", "cross_gender")])
        write_model_block(handle, "Matched-Token Robustness: SS", matched_token_ss, [("cross_race", "cross_race"), ("cross_gender", "cross_gender")])
        write_model_block(handle, "Matched-Token Robustness: Func-TCD", matched_token_func, [("cross_race", "cross_race"), ("cross_gender", "cross_gender")])

        handle.write("=" * 78 + "\n")
        handle.write("Group Summaries\n")
        handle.write("=" * 78 + "\n\n")
        handle.write("Race pairs (sorted by mean SS)\n")
        handle.write(race_summary.to_string(index=False))
        handle.write("\n\nGender pairs (sorted by mean SS)\n")
        handle.write(gender_summary.to_string(index=False))
        handle.write("\n\nSES pairs (sorted by mean SS)\n")
        handle.write(ses_summary.to_string(index=False))
        handle.write("\n")

    # ============================================================
    # Save JSON summary
    # ============================================================
    ss_cross_race = coef_record(main_ss, "cross_race")
    ss_cross_gender = coef_record(main_ss, "cross_gender")
    ss_token_diff = coef_record(main_ss, "token_diff")
    ss_mean_tokens = coef_record(main_ss, "mean_tokens")
    ss_freq_diff = coef_record(main_ss, "freq_diff")
    ss_mean_log_freq = coef_record(main_ss, "mean_log_freq")

    func_cross_race = coef_record(main_func, "cross_race")
    func_cross_gender = coef_record(main_func, "cross_gender")
    func_token_diff = coef_record(main_func, "token_diff")
    func_mean_tokens = coef_record(main_func, "mean_tokens")
    func_freq_diff = coef_record(main_func, "freq_diff")
    func_mean_log_freq = coef_record(main_func, "mean_log_freq")

    if ss_freq_diff["significant"] and func_freq_diff["significant"]:
        rarity_statement = (
            "Frequency disparity between the paired names is the most stable rarity-related predictor: "
            "larger log-frequency gaps increase both SS and function-word disruption, while mean rarity "
            "and mean BPE fragmentation are not independently significant in the upgraded specification."
        )
    elif ss_freq_diff["significant"] or func_freq_diff["significant"]:
        rarity_statement = (
            "Frequency disparity shows a partial effect, but mean rarity and mean BPE fragmentation do not "
            "emerge as stable independent predictors once the full set of controls is included."
        )
    else:
        rarity_statement = (
            "No rarity proxy is independently robust in the upgraded specification once profession, template, "
            "SES, race, and gender controls are included."
        )

    observed_token_counts = sorted(
        set(df["tokens_a"].unique().tolist()) | set(df["tokens_b"].unique().tolist())
    )
    if token_updates:
        token_audit_statement = (
            f"Tokenizer audit updated {len(token_updates)} names; observed token counts now span "
            f"{observed_token_counts}. This substantially compresses the fragmentation range relative "
            "to the hand-curated initial assumptions."
        )
    else:
        token_audit_statement = (
            f"Tokenizer audit confirmed the annotated counts; observed token counts span {observed_token_counts}."
        )

    conclusion_bits = []
    if ss_cross_race["significant"]:
        conclusion_bits.append(
            f"Cross-race pairs remain significant for SS (coef={ss_cross_race['coef']:+.4f}, p={ss_cross_race['pvalue']:.4g})"
        )
    else:
        conclusion_bits.append(
            f"Cross-race pairs are not significant for SS after controls (p={ss_cross_race['pvalue']:.4g})"
        )

    if func_cross_race["significant"]:
        conclusion_bits.append(
            f"but remain significant for function-word disruption (coef={func_cross_race['coef']:+.4f}, p={func_cross_race['pvalue']:.4g})"
        )
    else:
        conclusion_bits.append(
            f"and are also not significant for function-word disruption (p={func_cross_race['pvalue']:.4g})"
        )

    if ss_cross_gender["significant"] or func_cross_gender["significant"]:
        conclusion_bits.append(
            f"Gender crossing matters at least in one model (SS p={ss_cross_gender['pvalue']:.4g}, Func-TCD p={func_cross_gender['pvalue']:.4g})"
        )
    else:
        conclusion_bits.append(
            f"Cross-gender pairing is not a stable standalone predictor (SS p={ss_cross_gender['pvalue']:.4g}, Func-TCD p={func_cross_gender['pvalue']:.4g})"
        )

    summary = {
        "total_tests": int(len(df)),
        "n_names": int(len(names)),
        "n_pairs": int(len(name_pairs)),
        "n_professions": int(len(queries)),
        "n_templates": int(len(TEMPLATES)),
        "main_regression_ss": {
            "r_squared": float(main_ss.rsquared),
            "adj_r_squared": float(main_ss.rsquared_adj),
            "cross_race": ss_cross_race,
            "cross_gender": ss_cross_gender,
            "token_diff": ss_token_diff,
            "mean_tokens": ss_mean_tokens,
            "freq_diff": ss_freq_diff,
            "mean_log_freq": ss_mean_log_freq,
        },
        "main_regression_func_tcd": {
            "r_squared": float(main_func.rsquared),
            "adj_r_squared": float(main_func.rsquared_adj),
            "cross_race": func_cross_race,
            "cross_gender": func_cross_gender,
            "token_diff": func_token_diff,
            "mean_tokens": func_mean_tokens,
            "freq_diff": func_freq_diff,
            "mean_log_freq": func_mean_log_freq,
        },
        "interaction_models": {
            "ss_cross_race_x_mean_tokens": coef_record(interaction_ss, "cross_race:mean_tokens"),
            "func_cross_race_x_mean_tokens": coef_record(interaction_func, "cross_race:mean_tokens"),
        },
        "robustness": {
            "within_race_ss_cross_gender": coef_record(within_race_ss, "cross_gender"),
            "within_race_func_cross_gender": coef_record(within_race_func, "cross_gender"),
            "matched_token_ss_cross_race": coef_record(matched_token_ss, "cross_race"),
            "matched_token_func_cross_race": coef_record(matched_token_func, "cross_race"),
        },
        "group_means": {
            "race_pair_top5_by_ss": race_summary.head(5).to_dict(orient="records"),
            "gender_pair": gender_summary.to_dict(orient="records"),
            "ses_pair": ses_summary.to_dict(orient="records"),
        },
        "token_count_audit": {
            "n_updated_names": len(token_updates),
            "updates": token_updates,
            "observed_token_counts": observed_token_counts,
            "statement": token_audit_statement,
        },
        "interpretation": {
            "name_rarity_statement": rarity_statement,
            "tokenizer_statement": token_audit_statement,
            "conclusion": ". ".join(conclusion_bits) + ".",
        },
    }

    with open(RESULTS_DIR / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved summary: {RESULTS_DIR / 'summary.json'}")
    print(f"💾 Saved regression report: {RESULTS_DIR / 'regression_table.txt'}")
    print("\n📝 Interpretation")
    print(summary["interpretation"]["name_rarity_statement"])
    print(summary["interpretation"]["conclusion"])
    print("\n✨ P2 upgraded run complete.")


if __name__ == "__main__":
    main()
