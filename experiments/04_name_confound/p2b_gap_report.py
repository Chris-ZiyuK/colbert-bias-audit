#!/usr/bin/env python3
"""Summarize coverage gaps in the tokenizer-verified P2b name matrix."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFIED_PATH = Path(
    os.getenv(
        "P2B_VERIFIED_PATH",
        str(ROOT / "data" / "audit_names" / "name_features_tokenizer_verified.json"),
    )
)
MATRIX_PATH = Path(
    os.getenv(
        "P2B_MATRIX_PATH",
        str(ROOT / "data" / "audit_names" / "p2b_verified_matrix.json"),
    )
)
REPORT_PATH = Path(
    os.getenv(
        "P2B_GAP_REPORT_PATH",
        str(ROOT / "results" / "p2_name_confound" / "p2b_gap_report.txt"),
    )
)
SUMMARY_PATH = Path(
    os.getenv(
        "P2B_GAP_SUMMARY_PATH",
        str(ROOT / "results" / "p2_name_confound" / "p2b_gap_summary.json"),
    )
)
REPORT_TITLE = os.getenv("P2B_GAP_TITLE", "P2b gap report")


def main() -> None:
    with VERIFIED_PATH.open() as f:
        names = json.load(f)["names"]

    with MATRIX_PATH.open() as f:
        families = json.load(f)["families"]

    cells = defaultdict(list)
    for n in names:
        key = (n["race"], n["gender"], n["actual_bpe_tokens"])
        cells[key].append(
            {
                "name": n["name"],
                "freq": n.get("freq_value_for_matching", n.get("census_freq_per_100k")),
            }
        )

    tokenization_candidates = []
    for family_name in ("tokenization", "absolute_rarity"):
        fam = families[family_name]
        tokenization_candidates.append(
            {
                "family": family_name,
                "n_pairs": fam["n_pairs"],
                "n_treatment_pairs": fam["n_treatment_pairs"],
                "n_control_pairs": fam["n_control_pairs"],
            }
        )

    missing_cells = []
    for race in sorted({n["race"] for n in names}):
        for gender in sorted({n["gender"] for n in names}):
            token_buckets = {
                tok: len(cells[(race, gender, tok)])
                for tok in sorted({n["actual_bpe_tokens"] for n in names})
                if len(cells[(race, gender, tok)]) > 0
            }
            if len(token_buckets) < 2:
                missing_cells.append(
                    {
                        "race": race,
                        "gender": gender,
                        "issue": "Only one tokenizer bucket available",
                        "token_buckets": token_buckets,
                    }
                )
            else:
                sparse = {tok: count for tok, count in token_buckets.items() if count < 2}
                if sparse:
                    missing_cells.append(
                        {
                            "race": race,
                            "gender": gender,
                            "issue": "Tokenizer bucket too sparse for robust matched pairs",
                            "token_buckets": token_buckets,
                        }
                    )

    recommendations = [
        "Priority 1: add non-White names that stay 2+ tokens under the actual ColBERT tokenizer while remaining close in frequency to existing same-race same-gender 1-token names.",
        "Priority 2: add same-race same-gender names with near-equal frequency but different token counts, so tokenization-treatment pairs exceed the current count of 4.",
        "Priority 3: add near-equal-frequency pairs spanning higher and lower mean rarity within the same race/gender/token-count cells, so absolute-rarity treatment pairs exceed the current count of 2.",
    ]

    summary = {
        "cell_counts": {
            f"{race}|{gender}|{tok}": len(items) for (race, gender, tok), items in sorted(cells.items())
        },
        "family_pair_counts": tokenization_candidates,
        "coverage_gaps": missing_cells,
        "recommendations": recommendations,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        REPORT_TITLE,
        "=" * 60,
        "",
        "Current tokenizer-verified cell coverage",
    ]
    for (race, gender, tok), items in sorted(cells.items()):
        pretty = ", ".join(f"{x['name']}({x['freq']})" for x in items)
        lines.append(f"- {race} / {gender} / {tok} token(s): {len(items)} -> {pretty}")

    lines.extend(["", "Critical family counts"])
    for item in tokenization_candidates:
        lines.append(
            f"- {item['family']}: total_pairs={item['n_pairs']}, "
            f"treatment_pairs={item['n_treatment_pairs']}, control_pairs={item['n_control_pairs']}"
        )

    lines.extend(["", "Coverage gaps"])
    for gap in missing_cells:
        lines.append(
            f"- {gap['race']} / {gap['gender']}: {gap['issue']} -> {gap['token_buckets']}"
        )

    lines.extend(["", "Acquisition priorities"])
    for rec in recommendations:
        lines.append(f"- {rec}")

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Saved P2b gap report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
