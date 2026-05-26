"""
Name pool management and annotation.

Sources:
  - Bertrand & Mullainathan (2004): Original audit study names
  - Rosenman et al. (2023): Harvard Dataverse first-name ethnicity database

Each name is annotated with:
  - race: categorical (white, black, hispanic, asian)
  - gender: categorical (male, female)
  - bpe_tokens: int, number of BPE subword tokens
  - frequency: float, relative within-race frequency
  - rarity_class: categorical (common, rare) based on BPE token count
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class Name:
    """Annotated name for counterfactual auditing."""
    name: str
    race: str
    gender: str
    bpe_tokens: int = 1
    frequency: float = 0.0
    rarity_class: str = "common"  # 'common' if bpe_tokens == 1, else 'rare'

    def __post_init__(self):
        if self.rarity_class == "":
            self.rarity_class = "rare" if self.bpe_tokens > 1 else "common"


@dataclass
class NamePair:
    """A counterfactual name pair with matched-pair family annotations."""
    name_a: Name
    name_b: Name
    pair_type: str = ""  # e.g., 'cross_race', 'cross_gender', 'token_gap', etc.

    @property
    def same_race(self) -> bool:
        return self.name_a.race == self.name_b.race

    @property
    def same_gender(self) -> bool:
        return self.name_a.gender == self.name_b.gender

    @property
    def token_gap(self) -> int:
        return abs(self.name_a.bpe_tokens - self.name_b.bpe_tokens)

    @property
    def rarity_category(self) -> str:
        """CC / CR / RR rarity classification."""
        r_a = self.name_a.rarity_class
        r_b = self.name_b.rarity_class
        if r_a == "common" and r_b == "common":
            return "CC"
        elif r_a == "rare" and r_b == "rare":
            return "RR"
        else:
            return "CR"

    def label(self) -> str:
        return f"{self.name_a.name} → {self.name_b.name}"


# ============================================================================
# Bertrand & Mullainathan (2004) canonical name set
# ============================================================================

BM_NAMES = {
    "white_female": ["Emily", "Anne", "Jill", "Allison", "Laurie",
                      "Sarah", "Meredith", "Carrie", "Kristen"],
    "white_male":   ["Todd", "Neil", "Geoffrey", "Brett", "Brendan",
                      "Greg", "Matthew", "Jay", "Brad"],
    "black_female": ["Lakisha", "Tanisha", "Latoya", "Kenya", "Latonya",
                      "Ebony", "Aisha", "Tamika"],
    "black_male":   ["Jamal", "Leroy", "Jermaine", "Rasheed", "Tremayne",
                      "Kareem", "Darnell", "Tyrone", "Hakim"],
}


def load_name_pool(
    pool_path: Optional[Path] = None,
    tokenizer=None,
) -> List[Name]:
    """Load annotated name pool.

    If pool_path is provided, loads from JSON. Otherwise, uses the
    built-in BM (2004) name set.

    If tokenizer is provided, computes BPE token counts automatically.

    Args:
        pool_path: Path to name pool JSON file.
        tokenizer: HuggingFace tokenizer for computing BPE token counts.

    Returns:
        List of Name objects with all annotations.
    """
    if pool_path and pool_path.exists():
        with open(pool_path) as f:
            data = json.load(f)
        names = []
        for entry in data.get("names", data):
            n = Name(
                name=entry["name"],
                race=entry.get("race", ""),
                gender=entry.get("gender", ""),
                bpe_tokens=entry.get("bpe_tokens", 1),
                frequency=entry.get("frequency", 0.0),
                rarity_class=entry.get("rarity_class", ""),
            )
            names.append(n)
    else:
        # Build from built-in BM names
        names = []
        for group_key, name_list in BM_NAMES.items():
            race, gender = group_key.rsplit("_", 1)
            for nm in name_list:
                names.append(Name(name=nm, race=race, gender=gender))

    # Compute BPE token counts if tokenizer available
    if tokenizer is not None:
        for n in names:
            n.bpe_tokens = len(
                tokenizer.encode(n.name, add_special_tokens=False)
            )
            n.rarity_class = "rare" if n.bpe_tokens > 1 else "common"

    return names


def build_all_pairs(names: List[Name]) -> List[NamePair]:
    """Generate all ordered pairs from name pool."""
    pairs = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                pairs.append(NamePair(name_a=a, name_b=b))
    return pairs


def build_matched_families(
    names: List[Name],
) -> Dict[str, List[NamePair]]:
    """Construct the 5 matched-pair families for confound decomposition.

    Families:
      1. cross_race: different race, same gender, same BPE count
      2. cross_gender: different gender, same race, same BPE count
      3. token_gap: same race, same gender, different BPE count
      4. frequency_gap: same race, same gender, same BPE, different freq
      5. absolute_rarity: same race, same gender, both rare (>2 BPE tokens)
    """
    families: Dict[str, List[NamePair]] = {
        "cross_race": [],
        "cross_gender": [],
        "token_gap": [],
        "frequency_gap": [],
        "absolute_rarity": [],
    }

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            # Cross-race: same gender, same BPE, different race
            if (a.gender == b.gender and a.bpe_tokens == b.bpe_tokens
                    and a.race != b.race):
                families["cross_race"].append(
                    NamePair(a, b, pair_type="cross_race")
                )
            # Cross-gender: same race, same BPE, different gender
            if (a.race == b.race and a.bpe_tokens == b.bpe_tokens
                    and a.gender != b.gender):
                families["cross_gender"].append(
                    NamePair(a, b, pair_type="cross_gender")
                )
            # Token gap: same race, same gender, different BPE
            if (a.race == b.race and a.gender == b.gender
                    and a.bpe_tokens != b.bpe_tokens):
                families["token_gap"].append(
                    NamePair(a, b, pair_type="token_gap")
                )
            # Frequency gap: same everything except frequency
            if (a.race == b.race and a.gender == b.gender
                    and a.bpe_tokens == b.bpe_tokens
                    and abs(a.frequency - b.frequency) > 0.1):
                families["frequency_gap"].append(
                    NamePair(a, b, pair_type="frequency_gap")
                )
            # Absolute rarity: same race, same gender, both rare
            if (a.race == b.race and a.gender == b.gender
                    and a.bpe_tokens > 2 and b.bpe_tokens > 2):
                families["absolute_rarity"].append(
                    NamePair(a, b, pair_type="absolute_rarity")
                )

    return families
