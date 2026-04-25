#!/usr/bin/env python3
"""Build a tokenizer-verified P2b name and pair matrix.

P2b formalizes the "verified matrix" idea:
1. Recompute actual tokenizer segmentation for every name.
2. Export tokenizer-verified name metadata.
3. Export contrast families with treatment/control pair sets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.audit.core import MODEL_NAME

INPUT_PATH = ROOT / "data" / "audit_names" / "name_features.json"
VERIFIED_NAMES_PATH = ROOT / "data" / "audit_names" / "name_features_tokenizer_verified.json"
PAIR_MATRIX_PATH = ROOT / "data" / "audit_names" / "p2b_verified_matrix.json"


def log10_freq(value: float) -> float:
    import math

    return round(math.log10(max(value, 0.1)), 6)


def pair_id(a: str, b: str) -> str:
    left, right = sorted([a, b])
    return f"{left}-{right}"


def family_specs():
    return {
        "race": {
            "description": "Same gender, same tokenizer count, small frequency gap; compare cross-race vs same-race pairs.",
            "eligibility": lambda a, b: a["gender"] == b["gender"]
            and a["actual_bpe_tokens"] == b["actual_bpe_tokens"]
            and abs(a["log_freq"] - b["log_freq"]) <= 0.25,
            "treatment": lambda a, b: a["race"] != b["race"],
            "control_label": "same_race",
            "treatment_label": "cross_race",
        },
        "gender": {
            "description": "Same race, same tokenizer count, small frequency gap; compare cross-gender vs same-gender pairs.",
            "eligibility": lambda a, b: a["race"] == b["race"]
            and a["actual_bpe_tokens"] == b["actual_bpe_tokens"]
            and abs(a["log_freq"] - b["log_freq"]) <= 0.25,
            "treatment": lambda a, b: a["gender"] != b["gender"],
            "control_label": "same_gender",
            "treatment_label": "cross_gender",
        },
        "tokenization": {
            "description": "Same race, same gender, small frequency gap; compare token-count mismatch vs token-count matched pairs.",
            "eligibility": lambda a, b: a["race"] == b["race"]
            and a["gender"] == b["gender"]
            and abs(a["log_freq"] - b["log_freq"]) <= 0.25,
            "treatment": lambda a, b: a["actual_bpe_tokens"] != b["actual_bpe_tokens"],
            "control_label": "token_matched",
            "treatment_label": "token_mismatch",
        },
        "frequency_gap": {
            "description": "Same race, same gender, same tokenizer count; compare high vs low frequency-gap pairs.",
            "eligibility": lambda a, b: a["race"] == b["race"]
            and a["gender"] == b["gender"]
            and a["actual_bpe_tokens"] == b["actual_bpe_tokens"],
            "treatment": lambda a, b: abs(a["log_freq"] - b["log_freq"]) > 0.25,
            "control_label": "low_freq_gap",
            "treatment_label": "high_freq_gap",
        },
        "absolute_rarity": {
            "description": "Same race, same gender, same tokenizer count, near-equal frequency; compare low-mean-rarity vs high-mean-rarity pairs.",
            "eligibility": lambda a, b: a["race"] == b["race"]
            and a["gender"] == b["gender"]
            and a["actual_bpe_tokens"] == b["actual_bpe_tokens"]
            and abs(a["log_freq"] - b["log_freq"]) <= 0.10,
            "treatment": None,  # assigned after median split within eligible set
            "control_label": "higher_mean_freq",
            "treatment_label": "lower_mean_freq",
        },
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with INPUT_PATH.open() as f:
        raw = json.load(f)

    names = []
    for item in raw["names"]:
        token_ids = tokenizer.encode(item["name"], add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        verified = dict(item)
        verified["manual_bpe_tokens"] = int(item["bpe_tokens"])
        verified["actual_bpe_tokens"] = len(token_ids)
        verified["tokenizer_tokens"] = tokens
        verified["token_count_corrected"] = bool(len(token_ids) != item["bpe_tokens"])
        verified["log_freq"] = log10_freq(item["census_freq_per_100k"])
        names.append(verified)

    names.sort(key=lambda x: (x["race"], x["gender"], x["actual_bpe_tokens"], -x["census_freq_per_100k"], x["name"]))

    with VERIFIED_NAMES_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "_doc": "Tokenizer-verified version of the P2 name feature matrix using the actual ColBERT tokenizer.",
                "model_name": MODEL_NAME,
                "n_names": len(names),
                "n_corrected_token_counts": sum(1 for n in names if n["token_count_corrected"]),
                "names": names,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    specs = family_specs()
    family_payload = {}

    for family_name, spec in specs.items():
        eligible_pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = names[i]
                b = names[j]
                if not spec["eligibility"](a, b):
                    continue

                entry = {
                    "pair_id": pair_id(a["name"], b["name"]),
                    "name_a": a["name"],
                    "name_b": b["name"],
                    "race_a": a["race"],
                    "race_b": b["race"],
                    "gender_a": a["gender"],
                    "gender_b": b["gender"],
                    "tokens_a": a["actual_bpe_tokens"],
                    "tokens_b": b["actual_bpe_tokens"],
                    "freq_a": a["census_freq_per_100k"],
                    "freq_b": b["census_freq_per_100k"],
                    "log_freq_a": a["log_freq"],
                    "log_freq_b": b["log_freq"],
                    "freq_diff": round(abs(a["log_freq"] - b["log_freq"]), 6),
                    "mean_log_freq": round((a["log_freq"] + b["log_freq"]) / 2, 6),
                }
                eligible_pairs.append(entry)

        if family_name == "absolute_rarity":
            if eligible_pairs:
                median_mean = sorted(p["mean_log_freq"] for p in eligible_pairs)[len(eligible_pairs) // 2]
            else:
                median_mean = 0.0
            for entry in eligible_pairs:
                entry["group"] = (
                    spec["treatment_label"] if entry["mean_log_freq"] < median_mean else spec["control_label"]
                )
                entry["treatment"] = int(entry["mean_log_freq"] < median_mean)
        else:
            for entry in eligible_pairs:
                treatment = spec["treatment"](
                    {"race": entry["race_a"], "gender": entry["gender_a"], "actual_bpe_tokens": entry["tokens_a"], "log_freq": entry["log_freq_a"]},
                    {"race": entry["race_b"], "gender": entry["gender_b"], "actual_bpe_tokens": entry["tokens_b"], "log_freq": entry["log_freq_b"]},
                )
                entry["group"] = spec["treatment_label"] if treatment else spec["control_label"]
                entry["treatment"] = int(treatment)

        family_payload[family_name] = {
            "description": spec["description"],
            "control_label": spec["control_label"],
            "treatment_label": spec["treatment_label"],
            "n_pairs": len(eligible_pairs),
            "n_treatment_pairs": sum(p["treatment"] for p in eligible_pairs),
            "n_control_pairs": sum(1 - p["treatment"] for p in eligible_pairs),
            "pairs": eligible_pairs,
        }

    with PAIR_MATRIX_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "_doc": "Tokenizer-verified pair matrix for P2b targeted validation.",
                "source_names": str(VERIFIED_NAMES_PATH.relative_to(ROOT)),
                "families": family_payload,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"✅ Saved tokenizer-verified names to {VERIFIED_NAMES_PATH}")
    print(f"✅ Saved P2b verified pair matrix to {PAIR_MATRIX_PATH}")


if __name__ == "__main__":
    main()
