"""Plot aggregate MaxSim argmax-alignment diagnostics.

This converts the argmax-alignment JSON rows into a compact mechanism figure
for the paper: a document-token transition heatmap plus token-type diagnostic
bars. It uses aggregate counts rather than a single cherry-picked example.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROWS = ROOT / "results" / "14_argmax_alignment" / "argmax_alignment_rows.json"
DEFAULT_OUT = ROOT / "paper" / "figures" / "fig8_argmax_mechanism.pdf"


ORDER = ["function", "content", "special"]
COLORS = {
    "function": "#c84b4b",
    "content": "#2f6f9f",
    "special": "#9a9a9a",
}


def rate(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarise(rows: list[dict]) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    transitions = Counter()
    by_query_type: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        by_query_type[row["query_token_type"]].append(row)
        if row["argmax_changed"]:
            transitions[(row["doc_token_type_a"], row["doc_token_type_b"])] += 1

    matrix = np.zeros((len(ORDER), len(ORDER)))
    for i, src in enumerate(ORDER):
        row_total = sum(transitions[(src, dst)] for dst in ORDER)
        for j, dst in enumerate(ORDER):
            matrix[i, j] = transitions[(src, dst)] / row_total if row_total else 0.0

    metrics = {}
    for typ in ["function", "content"]:
        vals = by_query_type[typ]
        metrics[typ] = {
            "mean_tcd": mean([r["abs_tcd"] for r in vals]),
            "argmax_change": rate([r["argmax_changed"] for r in vals]),
            "name_context_hit": rate([r["either_name_context_hit_w2"] for r in vals]),
        }

    return matrix, metrics


def plot(rows_path: Path, out_path: Path) -> None:
    rows = json.loads(rows_path.read_text())
    matrix, metrics = summarise(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.1, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.45], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(matrix, cmap="Blues", vmin=0, vmax=max(0.8, matrix.max()))
    ax0.set_xticks(range(len(ORDER)))
    ax0.set_xticklabels(["Function", "Content", "Special"], rotation=35, ha="right", fontsize=8)
    ax0.set_yticks(range(len(ORDER)))
    ax0.set_yticklabels(["Function", "Content", "Special"], fontsize=8)
    ax0.set_xlabel("Argmax token type after swap", fontsize=8)
    ax0.set_ylabel("Argmax token type before swap", fontsize=8)
    ax0.set_title("Changed-argmax transitions", fontsize=10, fontweight="bold")
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            ax0.text(j, i, f"{100 * matrix[i, j]:.0f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    ax1 = fig.add_subplot(gs[0, 1])
    metric_labels = ["Mean TCD", "Argmax\nchange", "Name±2\nhit"]
    func_values = [
        metrics["function"]["mean_tcd"],
        metrics["function"]["argmax_change"],
        metrics["function"]["name_context_hit"],
    ]
    cont_values = [
        metrics["content"]["mean_tcd"],
        metrics["content"]["argmax_change"],
        metrics["content"]["name_context_hit"],
    ]
    x = np.arange(len(metric_labels))
    width = 0.35
    bars_f = ax1.bar(x - width / 2, func_values, width, color=COLORS["function"], label="Function")
    bars_c = ax1.bar(x + width / 2, cont_values, width, color=COLORS["content"], label="Content")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=8)
    ax1.set_title("Function words: high TCD, not high direct matching", fontsize=10, fontweight="bold")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, max(func_values + cont_values) * 1.22)

    for bars, values in [(bars_f, func_values), (bars_c, cont_values)]:
        for bar, val in zip(bars, values):
            label = f"{val:.3f}" if val < 0.1 else f"{val:.2f}"
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ax1.get_ylim()[1] * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(
        "MaxSim Argmax Alignment Check",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot(args.rows, args.out)


if __name__ == "__main__":
    main()
