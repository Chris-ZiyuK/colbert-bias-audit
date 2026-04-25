#!/usr/bin/env python3
"""Build a Rosenman-backed expansion matrix for P2b.

This script keeps the original 30-name audit set, adds a conservative batch of
manually gender-curated Rosenman names, and maps all names onto a single
frequency scale using Rosenman P(name | race) prevalence.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]

BASE_NAMES_PATH = ROOT / "data" / "audit_names" / "name_features.json"
ADDITIONS_PATH = ROOT / "data" / "audit_names" / "p2b_rosenman_additions.json"
RACE_PROB_PATH = ROOT / "data" / "audit_names" / "external" / "first_nameRaceProbs.tab"
PREVALENCE_PATH = ROOT / "data" / "audit_names" / "external" / "first_raceNameProbs.tab"

FEATURES_OUT_PATH = ROOT / "data" / "audit_names" / "p2b_rosenman_name_features.json"
MATRIX_OUT_PATH = ROOT / "data" / "audit_names" / "p2b_rosenman_matrix.json"
REPORT_OUT_PATH = ROOT / "results" / "p2_name_confound" / "p2b_rosenman_build_report.txt"

MODEL_NAME = "colbert-ir/colbertv2.0"

RACE_TO_KEY = {
    "White": "whi",
    "Black": "bla",
    "Hispanic": "his",
    "Asian": "asi",
}
KEY_TO_RACE = {v: k for k, v in RACE_TO_KEY.items()}


def normalize_name(name: str) -> str:
    return name.upper().replace("-", "").replace(" ", "")


def load_tab(path: Path) -> dict[str, dict[str, float]]:
    payload = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            payload[normalize_name(row["name"])] = {
                "whi": float(row["whi"]),
                "bla": float(row["bla"]),
                "his": float(row["his"]),
                "asi": float(row["asi"]),
                "oth": float(row["oth"]),
            }
    return payload


def pair_id(a: str, b: str) -> str:
    left, right = sorted([a, b])
    return f"{left}-{right}"


def family_specs():
    return {
        "race": {
            "description": "Same gender, same tokenizer count, small frequency gap; compare cross-race vs same-race pairs.",
            "eligibility": lambda a, b: a["gender"] == b["gender"]
            and a["actual_bpe_tokens"] == b["actual_bpe_tokens"]
            and abs(a["log_freq"] - b["log_freq"]) <= 0.35,
            "treatment": lambda a, b: a["race"] != b["race"],
            "control_label": "same_race",
            "treatment_label": "cross_race",
        },
        "gender": {
            "description": "Same race, same tokenizer count, small frequency gap; compare cross-gender vs same-gender pairs.",
            "eligibility": lambda a, b: a["race"] == b["race"]
            and a["actual_bpe_tokens"] == b["actual_bpe_tokens"]
            and abs(a["log_freq"] - b["log_freq"]) <= 0.35,
            "treatment": lambda a, b: a["gender"] != b["gender"],
            "control_label": "same_gender",
            "treatment_label": "cross_gender",
        },
        "tokenization": {
            "description": "Same race, same gender, small frequency gap; compare token-count mismatch vs token-count matched pairs.",
            "eligibility": lambda a, b: a["race"] == b["race"]
            and a["gender"] == b["gender"]
            and abs(a["log_freq"] - b["log_freq"]) <= 0.35,
            "treatment": lambda a, b: a["actual_bpe_tokens"] != b["actual_bpe_tokens"],
            "control_label": "token_matched",
            "treatment_label": "token_mismatch",
        },
        "frequency_gap": {
            "description": "Same race, same gender, same tokenizer count; compare high vs low frequency-gap pairs.",
            "eligibility": lambda a, b: a["race"] == b["race"]
            and a["gender"] == b["gender"]
            and a["actual_bpe_tokens"] == b["actual_bpe_tokens"],
            "treatment": lambda a, b: abs(a["log_freq"] - b["log_freq"]) > 0.35,
            "control_label": "low_freq_gap",
            "treatment_label": "high_freq_gap",
        },
        "absolute_rarity": {
            "description": "Same race, same gender, same tokenizer count, near-equal frequency; compare low-mean-rarity vs high-mean-rarity pairs.",
            "eligibility": lambda a, b: a["race"] == b["race"]
            and a["gender"] == b["gender"]
            and a["actual_bpe_tokens"] == b["actual_bpe_tokens"]
            and abs(a["log_freq"] - b["log_freq"]) <= 0.12,
            "treatment": None,
            "control_label": "higher_mean_freq",
            "treatment_label": "lower_mean_freq",
        },
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    race_prob = load_tab(RACE_PROB_PATH)
    prevalence = load_tab(PREVALENCE_PATH)

    base_names = json.loads(BASE_NAMES_PATH.read_text())["names"]
    additions = json.loads(ADDITIONS_PATH.read_text())["names"]

    names = []
    seen = set()
    missing_from_rosenman = []

    for item in base_names + additions:
        name = item["name"]
        key = normalize_name(name)
        if key in seen:
            continue
        seen.add(key)

        race_key = RACE_TO_KEY[item["race"]]
        prob_row = race_prob.get(key)
        prev_row = prevalence.get(key)

        if prob_row is None or prev_row is None:
            missing_from_rosenman.append(name)
            freq_value = float(item.get("census_freq_per_100k", 1.0))
            assigned_prob = None
            argmax_race = None
            argmax_prob = None
            freq_source = "census_fallback"
        else:
            freq_value = max(prev_row[race_key] * 100000, 0.01)
            assigned_prob = prob_row[race_key]
            best_key = max(prob_row, key=prob_row.get)
            argmax_race = KEY_TO_RACE.get(best_key, "Other")
            argmax_prob = prob_row[best_key]
            freq_source = "rosenman_p_name_given_race_per_100k"

        token_ids = tokenizer.encode(name, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        enriched = dict(item)
        enriched["source_id"] = item.get("source_id", "bm2004_curated_local" if item in base_names else "rosenman_olivella_imai_2023_first_names")
        enriched["manual_bpe_tokens"] = item.get("bpe_tokens")
        enriched["actual_bpe_tokens"] = len(token_ids)
        enriched["tokenizer_tokens"] = tokens
        enriched["token_count_corrected"] = item.get("bpe_tokens") is not None and len(token_ids) != item["bpe_tokens"]
        enriched["rosenman_prob_assigned_race"] = assigned_prob
        enriched["rosenman_argmax_race"] = argmax_race
        enriched["rosenman_argmax_prob"] = argmax_prob
        enriched["rosenman_prevalence_per_100k"] = round(freq_value, 6)
        enriched["freq_value_for_matching"] = round(freq_value, 6)
        enriched["freq_value_source"] = freq_source
        enriched["log_freq"] = round(math.log10(max(freq_value, 0.01)), 6)
        names.append(enriched)

    names.sort(
        key=lambda x: (
            x["race"],
            x["gender"],
            x["actual_bpe_tokens"],
            -x["freq_value_for_matching"],
            x["name"],
        )
    )

    FEATURES_OUT_PATH.write_text(
        json.dumps(
            {
                "_doc": (
                    "Rosenman-backed P2b name feature matrix. Frequency matching is performed "
                    "using Rosenman P(name | race) prevalence scaled to per-100k within the assigned race. "
                    "The original 30 audit names are retained as anchors, and conservative additions are appended."
                ),
                "model_name": MODEL_NAME,
                "source_tables": {
                    "race_given_name": str(RACE_PROB_PATH.relative_to(ROOT)),
                    "name_given_race": str(PREVALENCE_PATH.relative_to(ROOT)),
                    "base_names": str(BASE_NAMES_PATH.relative_to(ROOT)),
                    "additions": str(ADDITIONS_PATH.relative_to(ROOT)),
                },
                "n_names": len(names),
                "n_missing_from_rosenman": len(missing_from_rosenman),
                "missing_from_rosenman": missing_from_rosenman,
                "names": names,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
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
                    "freq_a": a["freq_value_for_matching"],
                    "freq_b": b["freq_value_for_matching"],
                    "log_freq_a": a["log_freq"],
                    "log_freq_b": b["log_freq"],
                    "freq_diff": round(abs(a["log_freq"] - b["log_freq"]), 6),
                    "mean_log_freq": round((a["log_freq"] + b["log_freq"]) / 2, 6),
                }
                eligible_pairs.append(entry)

        if family_name == "absolute_rarity":
            if eligible_pairs:
                med = sorted(p["mean_log_freq"] for p in eligible_pairs)[len(eligible_pairs) // 2]
            else:
                med = 0.0
            for entry in eligible_pairs:
                treatment = int(entry["mean_log_freq"] < med)
                entry["group"] = spec["treatment_label"] if treatment else spec["control_label"]
                entry["treatment"] = treatment
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

    MATRIX_OUT_PATH.write_text(
        json.dumps(
            {
                "_doc": "Tokenizer-verified Rosenman-backed pair matrix for expanded P2b targeted validation.",
                "source_names": str(FEATURES_OUT_PATH.relative_to(ROOT)),
                "families": family_payload,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    REPORT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "P2b Rosenman matrix build",
        "=" * 60,
        "",
        f"Names exported: {len(names)}",
        f"Missing from Rosenman tables: {len(missing_from_rosenman)}",
    ]
    if missing_from_rosenman:
        lines.append("Missing names: " + ", ".join(sorted(missing_from_rosenman)))
    lines.extend(["", "Family counts"])
    for family_name, family in family_payload.items():
        lines.append(
            f"- {family_name}: total_pairs={family['n_pairs']}, "
            f"treatment_pairs={family['n_treatment_pairs']}, control_pairs={family['n_control_pairs']}"
        )
    REPORT_OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"✅ Saved Rosenman-backed names to {FEATURES_OUT_PATH}")
    print(f"✅ Saved Rosenman-backed pair matrix to {MATRIX_OUT_PATH}")


if __name__ == "__main__":
    main()
