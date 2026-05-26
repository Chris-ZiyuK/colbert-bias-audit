"""
Experiment 9: Ranking Impact Study — Translating score-level TCD into
ranking-level consequences.

Addresses the key reviewer concern: "Score changed by 3%, but does ranking
actually change?"

Design:
  1. For each query, construct a pool of 100 candidate documents with varying
     identity markers and professional descriptions.
  2. Score all documents with ColBERTv2.
  3. Swap identity markers in target documents and re-score.
  4. Measure ranking-level metrics: rank flip rate, rank displacement,
     top-k exposure disparity, MRR change.
  5. Stratify by rarity category (CC/CR/RR) to test whether rare names
     suffer worse ranking outcomes.
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.colbert import ColBERTRetriever
from src.metrics.ranking import compute_ranking_metrics
from src.metrics.tcd import compute_score_sensitivity
from src.audit.names import load_name_pool, Name


# ==========================================================================
# Candidate pool construction
# ==========================================================================

TEMPLATES = [
    "{NAME} has over ten years of experience in this field.",
    "{NAME} is a dedicated professional with strong credentials.",
    "{NAME} brings extensive expertise and proven track record.",
    "{NAME} holds advanced certifications in their area of practice.",
    "{NAME} has worked with leading organisations across the industry.",
    "{NAME} is known for their innovative approach to problem-solving.",
    "{NAME} has published widely and received professional recognition.",
    "{NAME} maintains the highest standards of professional conduct.",
    "{NAME} graduated from a top-ranked program with distinction.",
    "{NAME} has consistently delivered outstanding results in their role.",
]


def build_candidate_pool(
    names: List[Name],
    pool_size: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """Build a diverse candidate pool for ranking evaluation.

    Creates documents using different templates and names from the pool,
    ensuring demographic diversity.
    """
    rng = random.Random(seed)
    pool = []

    for i in range(pool_size):
        name = rng.choice(names)
        template = rng.choice(TEMPLATES)
        doc = template.replace("{NAME}", name.name)
        pool.append({
            "doc_id": i,
            "document": doc,
            "name": name.name,
            "race": name.race,
            "gender": name.gender,
            "bpe_tokens": name.bpe_tokens,
            "rarity_class": name.rarity_class,
            "template_idx": TEMPLATES.index(template),
        })

    return pool


def run_ranking_impact(
    model: ColBERTRetriever,
    queries: List[str],
    name_pairs: List[Tuple[Name, Name]],
    pool_size: int = 100,
    k: int = 10,
    output_dir: Path = Path("results"),
    seed: int = 42,
) -> pd.DataFrame:
    """Run ranking impact analysis.

    For each (query, name_pair):
      1. Build candidate pool with name_a's document as target
      2. Score entire pool → get original rank of target
      3. Swap target's name to name_b → re-score target only
      4. Compute ranking metrics
    """
    names_list = load_name_pool()
    results = []

    for query in tqdm(queries, desc="Ranking impact"):
        # Build one candidate pool per query
        pool = build_candidate_pool(names_list, pool_size=pool_size, seed=seed)

        # Score all pool documents
        pool_scores = np.array([
            model.score(query, doc["document"]).total_score
            for doc in pool
        ])

        for name_a, name_b in name_pairs:
            # Insert target document with name_a
            target_template = TEMPLATES[0]
            target_doc_a = target_template.replace("{NAME}", name_a.name)
            target_doc_b = target_template.replace("{NAME}", name_b.name)

            # Score target documents
            score_a = model.score(query, target_doc_a).total_score
            score_b = model.score(query, target_doc_b).total_score

            # Create augmented score arrays
            scores_with_a = np.append(pool_scores, score_a)
            scores_with_b = np.append(pool_scores, score_b)
            target_idx = len(pool_scores)  # last position

            # Compute ranking metrics
            metrics = compute_ranking_metrics(
                scores_with_a, scores_with_b, target_idx, k=k
            )

            # Rarity category
            rarity_a = "rare" if name_a.bpe_tokens > 1 else "common"
            rarity_b = "rare" if name_b.bpe_tokens > 1 else "common"
            if rarity_a == "common" and rarity_b == "common":
                rarity_cat = "CC"
            elif rarity_a == "rare" and rarity_b == "rare":
                rarity_cat = "RR"
            else:
                rarity_cat = "CR"

            row = {
                "query": query,
                "name_a": name_a.name,
                "name_b": name_b.name,
                "race_a": name_a.race,
                "race_b": name_b.race,
                "rarity_category": rarity_cat,
                "bpe_a": name_a.bpe_tokens,
                "bpe_b": name_b.bpe_tokens,
                "score_a": score_a,
                "score_b": score_b,
                "ss": compute_score_sensitivity(score_a, score_b),
                **metrics,
            }
            results.append(row)

    df = pd.DataFrame(results)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "ranking_impact_results.csv", index=False)

    # Summary
    print("\n=== Ranking Impact Summary ===")
    print(f"Total tests: {len(df)}")
    print(f"Rank flip rate: {df['rank_flip'].mean():.1%}")
    print(f"Mean rank displacement: {df['rank_displacement'].mean():.2f}")
    print(f"Mean MRR change: {df['mrr_change'].mean():.4f}")

    # By rarity
    print("\nBy Rarity Category:")
    rarity_summary = df.groupby("rarity_category").agg(
        flip_rate=("rank_flip", "mean"),
        mean_displacement=("rank_displacement", "mean"),
        mean_mrr_change=("mrr_change", "mean"),
        n=("rank_flip", "count"),
    )
    print(rarity_summary.to_string())
    rarity_summary.to_csv(output_dir / "ranking_by_rarity.csv")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 9: Ranking Impact Study"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "09_ranking_impact",
    )
    parser.add_argument(
        "--professions-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "professions" / "professions.json",
    )
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-professions", type=int, default=10)
    args = parser.parse_args()

    model = ColBERTRetriever()
    model.load()

    with open(args.professions_json) as f:
        professions = json.load(f)["professions"]
    queries = [p["query"] for p in professions[:args.n_professions]]

    # Name pairs with annotations
    name_pairs = [
        (Name("Emily", "white", "female", 1), Name("Lakisha", "black", "female", 3)),
        (Name("Greg", "white", "male", 1), Name("Jamal", "black", "male", 2)),
        (Name("Sarah", "white", "female", 1), Name("Tanisha", "black", "female", 3)),
        (Name("Todd", "white", "male", 1), Name("Tyrone", "black", "male", 2)),
        (Name("Brad", "white", "male", 1), Name("Darnell", "black", "male", 2)),
    ]

    run_ranking_impact(
        model, queries, name_pairs,
        pool_size=args.pool_size, k=args.k,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
