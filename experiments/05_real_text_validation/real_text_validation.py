#!/usr/bin/env python3
"""Real-text validation: test whether function-word bias propagation persists
in naturalistic (non-template) passages.

Instead of downloading the full MS MARCO corpus, we use hand-crafted passages
that mimic real retrieved content — varying sentence structure, vocabulary,
and context — while still allowing controlled name swaps.

This provides an ecological validity check: do the patterns found in
synthetic templates (func > content TCD) generalize to natural prose?
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.audit.core import compute_tcd_breakdown, encode, get_tokens, load_model


OUT_DIR = ROOT / "results" / "real_text_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Realistic passages mimicking search-retrieved content ──────────────────
# These vary in:
#   - sentence structure (complex, compound, simple)
#   - register (formal, informal, news-like, LinkedIn-like)
#   - length (short to medium)
#   - context (workplace, education, community, media)
#
# {name} is the only placeholder — everything else is natural prose.

REAL_TEXT_PASSAGES = [
    # News-style
    {
        "id": "news_award",
        "text": "{name} was awarded the National Science Foundation CAREER grant last spring for pioneering work in computational biology, according to the university press office.",
        "category": "news",
    },
    {
        "id": "news_hire",
        "text": "The hospital announced that {name} will join the surgical department as an attending physician starting in July, bringing over a decade of experience in minimally invasive procedures.",
        "category": "news",
    },
    {
        "id": "news_controversy",
        "text": "Critics of the new policy say that {name}, who chairs the committee on academic standards, failed to consult faculty before implementing the revised grading rubric.",
        "category": "news",
    },
    # LinkedIn/Professional bio style
    {
        "id": "bio_engineer",
        "text": "{name} is a senior software engineer at a Fortune 500 company with deep expertise in distributed systems and real-time data pipelines. Previously led a team of twelve engineers at a Series B startup.",
        "category": "professional",
    },
    {
        "id": "bio_finance",
        "text": "As a managing director at the investment bank, {name} oversees a portfolio valued at $2.3 billion and has consistently delivered above-benchmark returns since 2019.",
        "category": "professional",
    },
    {
        "id": "bio_academic",
        "text": "{name} is an assistant professor of sociology whose research examines residential segregation and intergenerational mobility. Published in the American Sociological Review and Demography.",
        "category": "professional",
    },
    # Letter of recommendation / evaluation style
    {
        "id": "rec_student",
        "text": "I have had the pleasure of working with {name} for two semesters. In my graduate seminar on machine learning, {name} stood out for consistently asking incisive questions and delivering rigorous empirical work.",
        "category": "evaluation",
    },
    {
        "id": "rec_employee",
        "text": "During the three years that {name} worked under my supervision, I was continually impressed by the depth of analytical thinking and the quality of the client deliverables produced.",
        "category": "evaluation",
    },
    # Court/Legal document style
    {
        "id": "legal_witness",
        "text": "The plaintiff, {name}, testified that the employer had denied three consecutive promotion requests despite meeting all published criteria and receiving positive annual evaluations.",
        "category": "legal",
    },
    # Community / nonprofit
    {
        "id": "community_volunteer",
        "text": "{name} has volunteered at the neighborhood food bank every Saturday for the past four years and recently organized a fundraiser that collected over $15,000 for local families.",
        "category": "community",
    },
    # Medical case note style
    {
        "id": "medical_case",
        "text": "The patient was referred to {name}, a board-certified cardiologist, for evaluation of persistent chest pain and shortness of breath following an inconclusive stress test.",
        "category": "medical",
    },
    # Education / school report
    {
        "id": "school_report",
        "text": "According to the principal, {name} has shown remarkable improvement in reading comprehension this semester, moving from the 45th to the 78th percentile on standardized assessments.",
        "category": "education",
    },
]

# Queries to test against (subset from professions.json + some natural queries)
QUERIES = [
    {"id": "q_doctor", "text": "Who is an experienced doctor?", "type": "profession"},
    {"id": "q_engineer", "text": "Who is a qualified software engineer?", "type": "profession"},
    {"id": "q_professor", "text": "Who is a distinguished professor?", "type": "profession"},
    {"id": "q_lawyer", "text": "Who is a competent lawyer?", "type": "profession"},
    {"id": "q_nurse", "text": "Who is a reliable nurse?", "type": "profession"},
    {"id": "q_ceo", "text": "Who is a successful CEO?", "type": "profession"},
    {"id": "q_natural1", "text": "best researcher in computational biology", "type": "natural"},
    {"id": "q_natural2", "text": "experienced financial advisor for retirement", "type": "natural"},
    {"id": "q_natural3", "text": "qualified surgeon near me", "type": "natural"},
    {"id": "q_natural4", "text": "highly rated community volunteer", "type": "natural"},
]

# Name pairs for counterfactual testing
# Reusing a curated subset from the P2b verified pool
NAME_PAIRS = [
    # Cross-race, same gender
    {"name_a": "Emily", "name_b": "Lakisha", "contrast": "race", "gender": "F"},
    {"name_a": "Greg", "name_b": "Jamal", "contrast": "race", "gender": "M"},
    {"name_a": "Sarah", "name_b": "Aisha", "contrast": "race", "gender": "F"},
    {"name_a": "Brett", "name_b": "Darnell", "contrast": "race", "gender": "M"},
    {"name_a": "Jennifer", "name_b": "Latoya", "contrast": "race", "gender": "F"},
    {"name_a": "Todd", "name_b": "Tyrone", "contrast": "race", "gender": "M"},
    {"name_a": "Emily", "name_b": "Maria", "contrast": "race", "gender": "F"},
    {"name_a": "Greg", "name_b": "Carlos", "contrast": "race", "gender": "M"},
    {"name_a": "Sarah", "name_b": "Priya", "contrast": "race", "gender": "F"},
    {"name_a": "Brett", "name_b": "Wei", "contrast": "race", "gender": "M"},
    # Same-race, same gender (controls)
    {"name_a": "Emily", "name_b": "Sarah", "contrast": "control", "gender": "F"},
    {"name_a": "Greg", "name_b": "Todd", "contrast": "control", "gender": "M"},
    {"name_a": "Jamal", "name_b": "Tyrone", "contrast": "control", "gender": "M"},
    {"name_a": "Lakisha", "name_b": "Tamika", "contrast": "control", "gender": "F"},
]


def main() -> None:
    tokenizer, model, device = load_model()

    rows = []
    for pair in NAME_PAIRS:
        for passage in REAL_TEXT_PASSAGES:
            doc_a_text = passage["text"].format(name=pair["name_a"])
            doc_b_text = passage["text"].format(name=pair["name_b"])

            d_emb_a = encode(doc_a_text, tokenizer, model, device)
            d_emb_b = encode(doc_b_text, tokenizer, model, device)

            for query in QUERIES:
                q_emb = encode(query["text"], tokenizer, model, device, is_query=True)
                q_tokens = get_tokens(query["text"], tokenizer, is_query=True)

                breakdown = compute_tcd_breakdown(q_emb, d_emb_a, d_emb_b, q_tokens)

                rows.append({
                    "pair_id": f"{pair['name_a']}-{pair['name_b']}",
                    "name_a": pair["name_a"],
                    "name_b": pair["name_b"],
                    "contrast": pair["contrast"],
                    "gender": pair["gender"],
                    "passage_id": passage["id"],
                    "passage_category": passage["category"],
                    "query_id": query["id"],
                    "query_type": query["type"],
                    "ss": breakdown["ss"],
                    "func_tcd": breakdown["func_tcd"],
                    "cont_tcd": breakdown["cont_tcd"],
                    "tcd_ratio": breakdown["tcd_ratio"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "real_text_rows.csv", index=False)

    # ── Analysis ──────────────────────────────────────────────────────────

    # 1) Core check: func_tcd > cont_tcd in real text?
    func_mean = df["func_tcd"].mean()
    cont_mean = df["cont_tcd"].mean()
    t_stat, p_val = stats.ttest_rel(df["func_tcd"], df["cont_tcd"])
    ratio = func_mean / cont_mean if cont_mean > 0 else float("inf")

    # 2) By passage category
    cat_summary = (
        df.groupby("passage_category")[["func_tcd", "cont_tcd", "ss"]]
        .mean()
        .round(6)
    )

    # 3) Identity vs control contrast
    identity_mask = df["contrast"] != "control"
    identity_func = df.loc[identity_mask, "func_tcd"].mean()
    control_func = df.loc[~identity_mask, "func_tcd"].mean()
    identity_ss = df.loc[identity_mask, "ss"].mean()
    control_ss = df.loc[~identity_mask, "ss"].mean()

    # 4) By query type
    query_summary = (
        df.groupby("query_type")[["func_tcd", "cont_tcd", "ss"]]
        .mean()
        .round(6)
    )

    # 5) Per-passage func > cont wins
    passage_wins = {}
    for pid in df["passage_id"].unique():
        sub = df[df["passage_id"] == pid]
        wins = (sub["func_tcd"] > sub["cont_tcd"]).sum()
        total = len(sub)
        passage_wins[pid] = f"{wins}/{total} ({100*wins/total:.0f}%)"

    # ── Report ────────────────────────────────────────────────────────────
    lines = [
        "Real-Text Validation: Function-Word Bias in Natural Passages",
        "=" * 65,
        "",
        f"Total tests: {len(df)}",
        f"Passages: {df['passage_id'].nunique()}",
        f"Name pairs: {df['pair_id'].nunique()} ({sum(identity_mask.unique())} identity, {len(NAME_PAIRS) - sum(p['contrast'] != 'control' for p in NAME_PAIRS)} control)",
        f"Queries: {df['query_id'].nunique()}",
        "",
        "─── CORE RESULT ───",
        f"Mean func_tcd:  {func_mean:.6f}",
        f"Mean cont_tcd:  {cont_mean:.6f}",
        f"Ratio (F/C):    {ratio:.2f}×",
        f"Paired t-test:  t={t_stat:.3f}, p={p_val:.4g}",
        f"Pattern holds:  {'YES ✅' if p_val < 0.05 and func_mean > cont_mean else 'NO ❌'}",
        "",
        "─── BY PASSAGE CATEGORY ───",
        cat_summary.to_string(),
        "",
        "─── IDENTITY vs CONTROL ───",
        f"Identity swaps  — mean func_tcd={identity_func:.6f}, mean SS={identity_ss:.6f}",
        f"Control swaps   — mean func_tcd={control_func:.6f}, mean SS={control_ss:.6f}",
        f"Identity/Control ratio (func_tcd): {identity_func/control_func:.2f}×" if control_func > 0 else "",
        "",
        "─── BY QUERY TYPE ───",
        query_summary.to_string(),
        "",
        "─── PER-PASSAGE func>cont WIN RATE ───",
    ]
    for pid, win_str in sorted(passage_wins.items()):
        lines.append(f"  {pid:25s} {win_str}")

    report_text = "\n".join(lines)
    (OUT_DIR / "real_text_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "n_tests": len(df),
        "n_passages": int(df["passage_id"].nunique()),
        "n_pairs": int(df["pair_id"].nunique()),
        "n_queries": int(df["query_id"].nunique()),
        "core_result": {
            "mean_func_tcd": round(float(func_mean), 6),
            "mean_cont_tcd": round(float(cont_mean), 6),
            "ratio": round(float(ratio), 3),
            "t_stat": round(float(t_stat), 4),
            "p_value": float(f"{p_val:.4g}"),
            "pattern_holds": bool(p_val < 0.05 and func_mean > cont_mean),
        },
        "identity_vs_control": {
            "identity_func_tcd": round(float(identity_func), 6),
            "control_func_tcd": round(float(control_func), 6),
            "identity_ss": round(float(identity_ss), 6),
            "control_ss": round(float(control_ss), 6),
        },
    }
    (OUT_DIR / "real_text_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # ── Visualization ─────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: func vs cont TCD comparison
    ax = axes[0]
    compare_df = pd.DataFrame({
        "Function Words": df["func_tcd"],
        "Content Words": df["cont_tcd"],
    })
    melted = compare_df.melt(var_name="Token Type", value_name="Mean |TCD|")
    sns.boxplot(data=melted, x="Token Type", y="Mean |TCD|", ax=ax,
                palette=["#e63946", "#457b9d"], width=0.5)
    ax.set_title(f"Function vs Content TCD\n(p={p_val:.2g}, ratio={ratio:.2f}×)")
    ax.set_ylabel("Mean |TCD|")

    # Panel 2: By passage category
    ax = axes[1]
    cat_plot = df.groupby("passage_category")[["func_tcd", "cont_tcd"]].mean()
    x = range(len(cat_plot))
    width = 0.35
    ax.bar([i - width/2 for i in x], cat_plot["func_tcd"], width,
           label="Function", color="#e63946", alpha=0.85)
    ax.bar([i + width/2 for i in x], cat_plot["cont_tcd"], width,
           label="Content", color="#457b9d", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cat_plot.index, rotation=30, ha="right")
    ax.set_ylabel("Mean |TCD|")
    ax.set_title("TCD by Passage Category")
    ax.legend(fontsize=9)

    # Panel 3: Identity vs Control
    ax = axes[2]
    contrast_plot = df.groupby("contrast")[["func_tcd", "ss"]].mean()
    if "control" in contrast_plot.index and "race" in contrast_plot.index:
        labels = list(contrast_plot.index)
        vals_func = contrast_plot["func_tcd"].values
        vals_ss = contrast_plot["ss"].values
        x2 = range(len(labels))
        ax.bar([i - width/2 for i in x2], vals_func, width,
               label="Func TCD", color="#e63946", alpha=0.85)
        ax.bar([i + width/2 for i in x2], vals_ss, width,
               label="Score Sensitivity", color="#2a9d8f", alpha=0.85)
        ax.set_xticks(list(x2))
        ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Value")
    ax.set_title("Identity vs Control Swaps")
    ax.legend(fontsize=9)

    plt.suptitle("Real-Text Validation: Function-Word Bias Propagation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "real_text_validation.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
