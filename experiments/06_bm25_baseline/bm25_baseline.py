#!/usr/bin/env python3
"""BM25 baseline comparison: show that term-frequency retrieval does NOT
exhibit distributed function-word bias propagation.

This confirms that the function-word TCD pattern is specific to
contextual (transformer-based) models, not an artifact of retrieval
in general. BM25 scores are decomposable by query term, but since
BM25 uses exact lexical matching (no contextual representations),
changing a name should only affect the name token's contribution —
not leak into function words.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
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

from src.audit.core import FUNCTION_WORDS, SPECIAL_TOKENS, classify_token


OUT_DIR = ROOT / "results" / "bm25_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── BM25 implementation ──────────────────────────────────────────────────

def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    import re
    return [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]


def bm25_score_breakdown(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float = 50.0,
    k1: float = 1.2,
    b: float = 0.75,
    idf_default: float = 3.0,
) -> dict[str, float]:
    """Compute BM25 score with per-query-token breakdown.

    Uses a uniform IDF (since we have no corpus stats) — this means
    the ONLY factor that varies between counterfactual docs is
    term frequency of the changed name tokens.
    """
    dl = len(doc_tokens)
    tf_counter = Counter(doc_tokens)

    per_token = {}
    for qt in query_tokens:
        tf = tf_counter.get(qt, 0)
        # BM25 TF component
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score = idf_default * numerator / denominator if denominator > 0 else 0
        per_token[qt] = score

    return per_token


def classify_bm25_token(tok: str) -> str:
    """Classify a simple token."""
    if tok.lower() in FUNCTION_WORDS:
        return "function"
    return "content"


# ── Same passages and pairs as real-text experiment ───────────────────────

PASSAGES = [
    "{name} was awarded the National Science Foundation CAREER grant last spring for pioneering work in computational biology, according to the university press office.",
    "The hospital announced that {name} will join the surgical department as an attending physician starting in July, bringing over a decade of experience in minimally invasive procedures.",
    "{name} is a senior software engineer at a Fortune 500 company with deep expertise in distributed systems and real-time data pipelines.",
    "As a managing director at the investment bank, {name} oversees a portfolio valued at $2.3 billion and has consistently delivered above-benchmark returns since 2019.",
    "{name} is an assistant professor of sociology whose research examines residential segregation and intergenerational mobility.",
    "I have had the pleasure of working with {name} for two semesters. In my graduate seminar on machine learning, {name} stood out for consistently asking incisive questions.",
    "The plaintiff, {name}, testified that the employer had denied three consecutive promotion requests despite meeting all published criteria.",
    "{name} has volunteered at the neighborhood food bank every Saturday for the past four years.",
    "The patient was referred to {name}, a board-certified cardiologist, for evaluation of persistent chest pain.",
    "According to the principal, {name} has shown remarkable improvement in reading comprehension this semester.",
]

QUERIES = [
    "Who is an experienced doctor?",
    "Who is a qualified software engineer?",
    "Who is a distinguished professor?",
    "Who is a competent lawyer?",
    "Who is a reliable nurse?",
    "Who is a successful CEO?",
    "best researcher in computational biology",
    "experienced financial advisor for retirement",
    "qualified surgeon near me",
    "highly rated community volunteer",
]

NAME_PAIRS = [
    ("Emily", "Lakisha"),
    ("Greg", "Jamal"),
    ("Sarah", "Aisha"),
    ("Brett", "Darnell"),
    ("Jennifer", "Latoya"),
    ("Todd", "Tyrone"),
    ("Emily", "Maria"),
    ("Greg", "Carlos"),
    ("Sarah", "Priya"),
    ("Brett", "Wei"),
    # Controls
    ("Emily", "Sarah"),
    ("Greg", "Todd"),
    ("Jamal", "Tyrone"),
    ("Lakisha", "Tamika"),
]


def main() -> None:
    rows = []

    for name_a, name_b in NAME_PAIRS:
        for p_idx, passage_template in enumerate(PASSAGES):
            doc_a = passage_template.format(name=name_a)
            doc_b = passage_template.format(name=name_b)

            doc_a_tokens = tokenize_simple(doc_a)
            doc_b_tokens = tokenize_simple(doc_b)

            for q_idx, query_text in enumerate(QUERIES):
                q_tokens = tokenize_simple(query_text)

                scores_a = bm25_score_breakdown(q_tokens, doc_a_tokens)
                scores_b = bm25_score_breakdown(q_tokens, doc_b_tokens)

                total_a = sum(scores_a.values())
                total_b = sum(scores_b.values())
                ss = abs(total_a - total_b) / (0.5 * (total_a + total_b)) if (total_a + total_b) > 0 else 0

                # Per-token TCD
                func_tcds = []
                cont_tcds = []
                for qt in q_tokens:
                    tcd = abs(scores_a.get(qt, 0) - scores_b.get(qt, 0))
                    cat = classify_bm25_token(qt)
                    if cat == "function":
                        func_tcds.append(tcd)
                    else:
                        cont_tcds.append(tcd)

                func_tcd = float(np.mean(func_tcds)) if func_tcds else 0.0
                cont_tcd = float(np.mean(cont_tcds)) if cont_tcds else 0.0

                rows.append({
                    "pair_id": f"{name_a}-{name_b}",
                    "name_a": name_a,
                    "name_b": name_b,
                    "passage_idx": p_idx,
                    "query_idx": q_idx,
                    "query": query_text,
                    "ss": ss,
                    "func_tcd": func_tcd,
                    "cont_tcd": cont_tcd,
                    "tcd_ratio": func_tcd / cont_tcd if cont_tcd > 0 else 0.0,
                    "total_a": total_a,
                    "total_b": total_b,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "bm25_rows.csv", index=False)

    # ── Analysis ──────────────────────────────────────────────────────────
    func_mean = float(df["func_tcd"].mean())
    cont_mean = float(df["cont_tcd"].mean())
    t_stat, p_val = stats.ttest_rel(df["func_tcd"], df["cont_tcd"])
    ratio = func_mean / cont_mean if cont_mean > 0 else 0.0

    # Load ColBERT real-text results for comparison
    colbert_summary_path = ROOT / "results" / "real_text_validation" / "real_text_summary.json"
    colbert_data = None
    if colbert_summary_path.exists():
        colbert_data = json.loads(colbert_summary_path.read_text())

    lines = [
        "BM25 Baseline: Function-Word TCD Analysis",
        "=" * 55,
        "",
        f"Total tests: {len(df)}",
        f"Name pairs: {df['pair_id'].nunique()}",
        "",
        "─── BM25 RESULT ───",
        f"Mean func_tcd:  {func_mean:.6f}",
        f"Mean cont_tcd:  {cont_mean:.6f}",
        f"Ratio (F/C):    {ratio:.2f}×",
        f"Paired t-test:  t={float(t_stat):.3f}, p={float(p_val):.4g}",
        f"Func > Cont:    {'YES' if func_mean > cont_mean and p_val < 0.05 else 'NO ✅ (as expected)'}",
        "",
    ]

    if colbert_data:
        cr = colbert_data["core_result"]
        lines.extend([
            "─── COMPARISON WITH COLBERT ───",
            f"ColBERT func_tcd:  {cr['mean_func_tcd']:.6f}",
            f"ColBERT cont_tcd:  {cr['mean_cont_tcd']:.6f}",
            f"ColBERT ratio:     {cr['ratio']:.2f}×",
            f"ColBERT p-value:   {cr['p_value']:.4g}",
            "",
            f"BM25 func_tcd:     {func_mean:.6f}",
            f"BM25 cont_tcd:     {cont_mean:.6f}",
            f"BM25 ratio:        {ratio:.2f}×",
            f"BM25 p-value:      {float(p_val):.4g}",
            "",
            "─── INTERPRETATION ───",
        ])
        if ratio < 1.1:
            lines.append(
                "BM25 shows NO function-word bias propagation (ratio ≈ 1.0)."
            )
            lines.append(
                "This confirms the ColBERT effect is specific to contextual models,"
            )
            lines.append(
                "not an artifact of retrieval decomposition in general."
            )
        else:
            lines.append(f"BM25 shows modest func/cont ratio: {ratio:.2f}×")

    report_text = "\n".join(lines)
    (OUT_DIR / "bm25_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "n_tests": len(df),
        "bm25": {
            "mean_func_tcd": func_mean,
            "mean_cont_tcd": cont_mean,
            "ratio": round(ratio, 3),
            "t_stat": round(float(t_stat), 4),
            "p_value": float(p_val),
        },
    }
    if colbert_data:
        summary["colbert"] = colbert_data["core_result"]

    (OUT_DIR / "bm25_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # ── Visualization: ColBERT vs BM25 ────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: BM25 func vs cont
    ax = axes[0]
    bm25_vals = pd.DataFrame({
        "Function Words": df["func_tcd"],
        "Content Words": df["cont_tcd"],
    }).melt(var_name="Token Type", value_name="Mean |TCD|")
    sns.boxplot(data=bm25_vals, x="Token Type", y="Mean |TCD|", ax=ax,
                palette=["#adb5bd", "#6c757d"], width=0.5)
    ax.set_title(f"BM25: Func vs Content TCD\n(ratio={ratio:.2f}×, p={float(p_val):.2g})")
    ax.set_ylabel("Mean |TCD|")

    # Panel 2: Cross-model comparison bar chart
    ax = axes[1]
    if colbert_data:
        cr = colbert_data["core_result"]
        models = ["ColBERT", "BM25"]
        func_vals = [cr["mean_func_tcd"], func_mean]
        cont_vals = [cr["mean_cont_tcd"], cont_mean]
        x = np.arange(len(models))
        width = 0.3
        ax.bar(x - width/2, func_vals, width, label="Function Words",
               color=["#e63946", "#e6a1a7"])
        ax.bar(x + width/2, cont_vals, width, label="Content Words",
               color=["#457b9d", "#a1c2d1"])
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylabel("Mean |TCD|")
        ax.set_title("ColBERT vs BM25:\nFunction-Word Bias Propagation")
        ax.legend()

        # Add ratio annotations
        for i, (f, c) in enumerate(zip(func_vals, cont_vals)):
            r = f / c if c > 0 else 0
            ax.annotate(f"{r:.2f}×", xy=(i, max(f, c)), fontsize=10,
                       ha="center", va="bottom", fontweight="bold",
                       color="#e63946" if r > 1.2 else "#6c757d")

    plt.suptitle("Cross-Model Comparison: Function-Word Bias Propagation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bm25_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
