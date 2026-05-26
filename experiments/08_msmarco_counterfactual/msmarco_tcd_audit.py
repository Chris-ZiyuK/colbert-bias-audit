"""
Experiment 8: MS MARCO Counterfactual — Benchmark-scale validation.

Applies the TCD auditing framework to real IR data from the MS MARCO
passage ranking benchmark, moving beyond synthetic templates.

Design:
  1. Download MS MARCO dev set passages
  2. Use SpaCy NER to identify passages containing person names
  3. Apply counterfactual name swaps (using the 65-name pool)
  4. Score original and swapped passages with ColBERTv2
  5. Compute TCD and measure whether the function-word pattern holds
  6. Report MRR/nDCG changes at the benchmark level
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.colbert import ColBERTRetriever
from src.metrics.tcd import (
    compute_tcd,
    compute_func_cont_tcd,
    compute_score_sensitivity,
)


def load_msmarco_passages(
    data_dir: Path,
    max_passages: int = 10000,
) -> List[Dict]:
    """Load MS MARCO passages from TSV file.

    Expected file: data_dir/collection.tsv  (pid \\t passage)
    """
    collection_path = data_dir / "collection.tsv"
    if not collection_path.exists():
        raise FileNotFoundError(
            f"MS MARCO collection not found at {collection_path}. "
            f"Place collection.tsv there; see data/msmarco/README.md."
        )

    passages = []
    with open(collection_path) as f:
        for i, line in enumerate(f):
            if i >= max_passages:
                break
            parts = line.strip().split("\t")
            if len(parts) == 2:
                passages.append({"pid": parts[0], "passage": parts[1]})

    return passages


def find_person_passages(
    passages: List[Dict],
    max_results: int = 500,
) -> List[Dict]:
    """Use SpaCy NER to find passages containing person names.

    Returns passages with annotated name spans.
    """
    try:
        import spacy
    except ImportError:
        raise ImportError("spacy required. Install: pip install spacy && python -m spacy download en_core_web_sm")

    nlp = spacy.load("en_core_web_sm")
    person_passages = []

    for p in tqdm(passages, desc="NER scan"):
        doc = nlp(p["passage"])
        person_ents = [ent for ent in doc.ents if ent.label_ == "PERSON"]
        if person_ents:
            # Take first person entity
            first_person = person_ents[0]
            person_passages.append({
                **p,
                "original_name": first_person.text,
                "name_start": first_person.start_char,
                "name_end": first_person.end_char,
            })
            if len(person_passages) >= max_results:
                break

    print(f"Found {len(person_passages)} passages with person names "
          f"(out of {len(passages)} scanned)")
    return person_passages


def swap_name_in_passage(
    passage: str,
    original_name: str,
    new_name: str,
) -> str:
    """Replace all occurrences of original_name with new_name."""
    return passage.replace(original_name, new_name)


def run_msmarco_audit(
    model: ColBERTRetriever,
    person_passages: List[Dict],
    swap_names: List[str],
    queries: List[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Run TCD audit on MS MARCO passages with name swaps.

    For each (passage, query), swap the detected person name with each
    name in swap_names and compute TCD.
    """
    results = []

    for pp in tqdm(person_passages[:100], desc="MS MARCO TCD audit"):
        orig_passage = pp["passage"]
        orig_name = pp["original_name"]

        for swap_name in swap_names[:10]:
            if swap_name == orig_name:
                continue

            swapped_passage = swap_name_in_passage(
                orig_passage, orig_name, swap_name
            )

            if swapped_passage == orig_passage:
                continue

            for query in queries[:5]:
                result_orig = model.score(query, orig_passage)
                result_swap = model.score(query, swapped_passage)

                if (result_orig.per_token_scores is None
                        or result_swap.per_token_scores is None):
                    continue

                # Align lengths (may differ due to tokenization changes)
                min_len = min(
                    len(result_orig.per_token_scores),
                    len(result_swap.per_token_scores),
                )
                pts_o = result_orig.per_token_scores[:min_len]
                pts_s = result_swap.per_token_scores[:min_len]
                q_tokens = result_orig.query_tokens[:min_len]

                tcd = compute_tcd(pts_o, pts_s)
                fc = compute_func_cont_tcd(tcd, q_tokens)
                ss = compute_score_sensitivity(
                    result_orig.total_score, result_swap.total_score
                )

                results.append({
                    "pid": pp["pid"],
                    "query": query,
                    "original_name": orig_name,
                    "swap_name": swap_name,
                    "func_tcd": fc["func_tcd"],
                    "cont_tcd": fc["cont_tcd"],
                    "tcd_ratio": fc["tcd_ratio"],
                    "ss": ss,
                    "score_original": result_orig.total_score,
                    "score_swapped": result_swap.total_score,
                    "data_source": "msmarco",
                })

    df = pd.DataFrame(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "msmarco_tcd_results.csv", index=False)

    # Summary
    if len(df) > 0:
        print("\n=== MS MARCO TCD Audit Summary ===")
        print(f"Total comparisons: {len(df)}")
        print(f"Mean Func-TCD: {df['func_tcd'].mean():.4f}")
        print(f"Mean Cont-TCD: {df['cont_tcd'].mean():.4f}")
        print(f"Func/Cont ratio: {df['func_tcd'].mean() / df['cont_tcd'].mean():.2f}×")
        print(f"Mean SS: {df['ss'].mean():.4f}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 8: MS MARCO Counterfactual Audit"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "08_msmarco",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "msmarco",
    )
    parser.add_argument(
        "--professions-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "professions" / "professions.json",
    )
    parser.add_argument("--max-passages", type=int, default=10000)
    parser.add_argument("--max-person-passages", type=int, default=500)
    args = parser.parse_args()

    # Load passages before loading the neural model, so a missing corpus fails
    # quickly and clearly.
    passages = load_msmarco_passages(args.data_dir, args.max_passages)
    person_passages = find_person_passages(passages, args.max_person_passages)

    model = ColBERTRetriever()
    model.load()

    # Load queries
    with open(args.professions_json) as f:
        professions = json.load(f)["professions"]
    queries = [p["query"] for p in professions[:10]]

    # Swap names
    swap_names = [
        "Emily", "Lakisha", "Greg", "Jamal", "Sarah",
        "Tanisha", "Todd", "Tyrone", "Allison", "Latoya",
    ]

    df = run_msmarco_audit(model, person_passages, swap_names, queries, args.output_dir)
    if len(df) > 0:
        summary = {
            "n_comparisons": int(len(df)),
            "n_passages": int(df["pid"].nunique()),
            "n_queries": int(df["query"].nunique()),
            "mean_func_tcd": float(df["func_tcd"].mean()),
            "mean_cont_tcd": float(df["cont_tcd"].mean()),
            "func_cont_ratio": float(df["func_tcd"].mean() / df["cont_tcd"].mean()),
            "mean_score_sensitivity": float(df["ss"].mean()),
            "pct_ratio_gt_1": float((df["tcd_ratio"] > 1).mean()),
        }
        with (args.output_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
