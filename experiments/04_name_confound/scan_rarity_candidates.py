#!/usr/bin/env python3
"""Scan Rosenman tables to find candidate names for absolute_rarity expansion.

Goal: find names within the same (race, gender, token_count) cells that have
similar log_freq (diff ≤ 0.12) but span a wider range of mean_log_freq values.
This closes the absolute_rarity gap in P2b.

Output: prints candidate names with their features, grouped by cell, and
shows which existing names they could pair with.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]

RACE_PROB_PATH = ROOT / "data" / "audit_names" / "external" / "first_nameRaceProbs.tab"
PREVALENCE_PATH = ROOT / "data" / "audit_names" / "external" / "first_raceNameProbs.tab"
FEATURES_PATH = ROOT / "data" / "audit_names" / "p2b_rosenman_name_features.json"

MODEL_NAME = "colbert-ir/colbertv2.0"

RACE_TO_KEY = {"White": "whi", "Black": "bla", "Hispanic": "his", "Asian": "asi"}
KEY_TO_RACE = {v: k for k, v in RACE_TO_KEY.items()}

# Manually curated gender lookup for common unambiguous first names
# Only include names with very clear conventional gender assignment
KNOWN_MALE = {
    "JAMES", "JOHN", "ROBERT", "MICHAEL", "WILLIAM", "DAVID", "RICHARD",
    "JOSEPH", "THOMAS", "CHARLES", "DANIEL", "MATTHEW", "ANTHONY", "MARK",
    "DONALD", "STEVEN", "PAUL", "ANDREW", "JOSHUA", "KENNETH", "KEVIN",
    "BRIAN", "GEORGE", "TIMOTHY", "RONALD", "EDWARD", "JASON", "JEFFREY",
    "RYAN", "JACOB", "GARY", "NICHOLAS", "ERIC", "JONATHAN", "STEPHEN",
    "LARRY", "JUSTIN", "SCOTT", "BRANDON", "BENJAMIN", "SAMUEL", "RAYMOND",
    "PATRICK", "JACK", "DENNIS", "JERRY", "TYLER", "AARON", "JOSE",
    "ADAM", "NATHAN", "HENRY", "DOUGLAS", "PETER", "ZACHARY", "KYLE",
    # Black male names
    "TERRENCE", "DEANDRE", "DESHAWN", "LAMAR", "MARQUIS", "TERRELL",
    "RASHAD", "CEDRIC", "REGINALD", "WILLIE", "CORNELIUS", "LEROY",
    "JEROME", "CLIFTON", "WARDELL", "DEXTER", "DWAYNE", "DARRYL",
    "LAMONT", "RODERICK", "KAREEM", "HAKEEM", "ANTWAN", "DEVONTE",
    # Hispanic male names
    "MIGUEL", "JORGE", "RICARDO", "FRANCISCO", "EDUARDO", "FERNANDO",
    "PEDRO", "RAFAEL", "OSCAR", "HECTOR", "ARMANDO", "ERNESTO",
    "SERGIO", "ROBERTO", "RAMIRO", "JESUS", "ALFREDO", "CARLOS",
    "GUILLERMO", "SALVADOR", "GERARDO", "ROGELIO", "GILBERTO",
    "REYNALDO", "ESTEBAN",
    # Asian male names
    "RAJESH", "SURESH", "DEEPAK", "ANIL", "RAVI", "SANJAY", "VIKRAM",
    "HARISH", "GANESH", "DINESH", "NIKHIL", "ROHIT", "ANAND",
    "ARJUN", "ASHOK", "MANOJ", "PRADEEP",
}

KNOWN_FEMALE = {
    "MARY", "PATRICIA", "JENNIFER", "LINDA", "BARBARA", "ELIZABETH",
    "SUSAN", "JESSICA", "SARAH", "KAREN", "LISA", "NANCY", "BETTY",
    "MARGARET", "SANDRA", "ASHLEY", "DOROTHY", "KIMBERLY", "EMILY",
    "DONNA", "MICHELLE", "CAROL", "AMANDA", "MELISSA", "DEBORAH",
    "STEPHANIE", "REBECCA", "SHARON", "LAURA", "CYNTHIA", "KATHLEEN",
    "AMY", "ANGELA", "SHIRLEY", "ANNA", "BRENDA", "PAMELA", "EMMA",
    "NICOLE", "HELEN", "SAMANTHA", "KATHERINE", "CHRISTINE", "DEBRA",
    "RACHEL", "CAROLYN", "JANET", "CATHERINE", "MARIA", "HEATHER",
    "DIANE", "RUTH", "JULIE", "OLIVIA", "JOYCE", "VIRGINIA",
    # Black female names
    "KEISHA", "SHANIQUA", "TANISHA", "TANESHA", "SHANICE", "PRECIOUS",
    "SHONDA", "LAKEISHA", "TOMIKA", "TAMEKA", "MONIQUE", "CHANTEL",
    "TIFFANY", "YOLANDA", "RAQUEL", "IMANI", "AALIYAH", "JANAE",
    # Hispanic female names
    "ROSA", "CARMEN", "ANA", "LUCIA", "ELENA", "PATRICIA", "ADRIANA",
    "CLAUDIA", "YOLANDA", "VERONICA", "LETICIA", "FRANCISCA",
    "MARGARITA", "ESPERANZA", "MARIBEL", "GRISELDA", "XIOMARA",
    # Asian female names
    "DEEPA", "ANITA", "SUNITA", "KAVITA", "SEEMA", "NISHA",
    "POOJA", "NEHA", "PADMA", "MEENA", "REKHA", "ASHA",
}


def normalize_name(name: str) -> str:
    return name.upper().replace("-", "").replace(" ", "")


def get_gender(name_key: str) -> str | None:
    if name_key in KNOWN_MALE:
        return "M"
    if name_key in KNOWN_FEMALE:
        return "F"
    return None


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


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    race_prob = load_tab(RACE_PROB_PATH)
    prevalence = load_tab(PREVALENCE_PATH)

    # Load existing names
    existing_features = json.loads(FEATURES_PATH.read_text())
    existing_names = {n["name"].upper() for n in existing_features["names"]}

    # Build lookup of existing names by (race, gender, token_count)
    existing_by_cell: dict[tuple, list[dict]] = {}
    for n in existing_features["names"]:
        cell = (n["race"], n["gender"], n["actual_bpe_tokens"])
        existing_by_cell.setdefault(cell, []).append(n)

    print("=" * 80)
    print("EXISTING NAMES BY CELL (race, gender, token_count)")
    print("=" * 80)
    for cell, names in sorted(existing_by_cell.items()):
        log_freqs = [n["log_freq"] for n in names]
        print(f"\n{cell}: {len(names)} names, log_freq range [{min(log_freqs):.2f}, {max(log_freqs):.2f}]")
        for n in sorted(names, key=lambda x: x["log_freq"]):
            print(f"  {n['name']:20s} log_freq={n['log_freq']:.3f}  tokens={n['actual_bpe_tokens']}")

    # Now scan Rosenman tables for candidates
    print("\n" + "=" * 80)
    print("SCANNING FOR RARITY EXPANSION CANDIDATES")
    print("=" * 80)

    MIN_RACE_PROB = 0.70  # Must be high-confidence for assigned race
    MAX_LOG_FREQ_DIFF = 0.12  # Same constraint as absolute_rarity family

    candidates_by_cell: dict[tuple, list[dict]] = {}

    for name_key, prob_row in race_prob.items():
        if name_key in existing_names:
            continue

        # Determine race assignment
        best_key = max(prob_row, key=lambda k: prob_row[k] if k != "oth" else 0)
        if best_key == "oth" or prob_row[best_key] < MIN_RACE_PROB:
            continue

        race = KEY_TO_RACE.get(best_key)
        if race is None:
            continue

        # Determine gender
        gender = get_gender(name_key)
        if gender is None:
            continue

        # Get prevalence
        prev_row = prevalence.get(name_key)
        if prev_row is None:
            continue

        freq_value = max(prev_row[best_key] * 100000, 0.01)
        log_freq = math.log10(max(freq_value, 0.01))

        # Get token count
        # Use title case for tokenization
        display_name = name_key.title()
        token_ids = tokenizer.encode(display_name, add_special_tokens=False)
        token_count = len(token_ids)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        cell = (race, gender, token_count)

        # Check if this could pair with any existing name in the same cell
        existing_in_cell = existing_by_cell.get(cell, [])
        if not existing_in_cell:
            continue

        # Find how many existing names it can pair with (log_freq diff <= 0.12)
        close_matches = []
        for ex in existing_in_cell:
            diff = abs(log_freq - ex["log_freq"])
            if diff <= MAX_LOG_FREQ_DIFF:
                close_matches.append((ex["name"], diff, ex["log_freq"]))

        if not close_matches:
            continue

        candidates_by_cell.setdefault(cell, []).append({
            "name": display_name,
            "race": race,
            "gender": gender,
            "token_count": token_count,
            "tokens": tokens,
            "race_prob": prob_row[best_key],
            "log_freq": round(log_freq, 4),
            "freq_per_100k": round(freq_value, 2),
            "close_matches": close_matches,
        })

    # Print candidates grouped by cell
    total_candidates = 0
    for cell, cands in sorted(candidates_by_cell.items()):
        cands.sort(key=lambda x: x["log_freq"])
        print(f"\n--- {cell} ---")
        print(f"  Existing: {[n['name'] for n in sorted(existing_by_cell.get(cell, []), key=lambda x: x['log_freq'])]}")
        for c in cands:
            total_candidates += 1
            match_str = ", ".join(f"{m[0]}(Δ={m[1]:.3f})" for m in c["close_matches"])
            print(
                f"  CANDIDATE: {c['name']:20s} log_freq={c['log_freq']:.3f}  "
                f"freq/100k={c['freq_per_100k']:>10.2f}  "
                f"race_prob={c['race_prob']:.3f}  "
                f"tokens={c['tokens']}  "
                f"matches=[{match_str}]"
            )

    print(f"\n\nTotal candidates found: {total_candidates}")

    # Now compute how many absolute_rarity pairs we could form with top picks
    # For each cell: pick names that maximize the spread of mean_log_freq
    # while still having pairwise log_freq diff <= 0.12
    print("\n" + "=" * 80)
    print("RECOMMENDED ADDITIONS (targeting max absolute_rarity pairs)")
    print("=" * 80)

    # Focus on cells that already have 2+ names (can already form pairs)
    # and cells where adding 1-2 names creates the most new pairs
    for cell, cands in sorted(candidates_by_cell.items()):
        existing_in_cell = existing_by_cell.get(cell, [])
        if len(existing_in_cell) < 1:
            continue

        # Show top 5 candidates per cell
        print(f"\n--- {cell} ({len(existing_in_cell)} existing) ---")
        for c in cands[:5]:
            match_str = ", ".join(f"{m[0]}(Δ={m[1]:.3f})" for m in c["close_matches"])
            print(
                f"  ★ {c['name']:20s} log_freq={c['log_freq']:.3f}  "
                f"race_prob={c['race_prob']:.3f}  "
                f"matches=[{match_str}]"
            )


if __name__ == "__main__":
    main()
