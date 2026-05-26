"""
Experiment 7b: Multi-Model Comparison — Cross-architecture bias analysis.

Compares bias across 5 retrieval architectures using the same counterfactual
design to establish whether the function-word propagation pattern is
ColBERT-specific or a general property of contextual models.

Models:
  1. ColBERTv2    — late-interaction, full token-level TCD
  2. DPR          — single-vector dense, aggregate score only
  3. SPLADE       — sparse neural, per-term weights
  4. CrossEncoder — joint encoding, aggregate score only
  5. Contriever   — contrastive dense, aggregate score only
  6. BM25         — lexical baseline (zero TCD expected)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.colbert import ColBERTRetriever
from src.models.dpr import DPRRetriever
from src.models.splade import SPLADERetriever
from src.models.cross_encoder import CrossEncoderRetriever
from src.models.contriever import ContrieverRetriever
from src.metrics.tcd import compute_score_sensitivity


def load_test_config(
    professions_json: Path,
    n_professions: int = 33,
) -> Tuple[List[str], List[Tuple[str, str]], str]:
    """Load queries, name pairs, and document template."""
    with open(professions_json) as f:
        professions = json.load(f)["professions"]
    queries = [p["query"] for p in professions[:n_professions]]

    name_pairs = [
        ("Emily", "Lakisha"),
        ("Greg", "Jamal"),
        ("Sarah", "Tanisha"),
        ("Todd", "Tyrone"),
        ("Allison", "Latoya"),
        ("Anne", "Aisha"),
        ("Brad", "Darnell"),
        ("Meredith", "Ebony"),
        ("Neil", "Rasheed"),
        ("Kristen", "Tamika"),
    ]

    template = "{NAME} has over ten years of experience in this field."

    return queries, name_pairs, template


def run_model_comparison(
    queries: List[str],
    name_pairs: List[Tuple[str, str]],
    template: str,
    output_dir: Path,
    skip_models: List[str] = None,
) -> pd.DataFrame:
    """Run all models on the same counterfactual pairs.

    For each model, compute:
      - Mean Score Sensitivity (SS)
      - Mean absolute score gap
      - Std of score gap
      - For ColBERT only: Func-TCD, Cont-TCD, ratio
    """
    skip_models = skip_models or []

    models = {
        "ColBERTv2": ColBERTRetriever(),
        "DPR": DPRRetriever(),
        "SPLADE": SPLADERetriever(),
        "CrossEncoder": CrossEncoderRetriever(),
        "Contriever": ContrieverRetriever(),
    }

    all_results = []

    for model_name, model in models.items():
        if model_name in skip_models:
            print(f"⏭ Skipping {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")

        try:
            model.load()
        except Exception as e:
            print(f"  ✗ Failed to load {model_name}: {e}")
            continue

        model_results = []

        for query in tqdm(queries, desc=f"{model_name}"):
            for name_a, name_b in name_pairs:
                doc_a = template.replace("{NAME}", name_a)
                doc_b = template.replace("{NAME}", name_b)

                cf = model.counterfactual_score(query, doc_a, doc_b)

                row = {
                    "model": model_name,
                    "query": query,
                    "name_a": name_a,
                    "name_b": name_b,
                    "score_a": cf.score_a,
                    "score_b": cf.score_b,
                    "score_gap": abs(cf.score_a - cf.score_b),
                    "ss": cf.score_sensitivity,
                    "supports_token_level": cf.supports_token_level,
                }

                # Token-level metrics for ColBERT
                if cf.supports_token_level and cf.tcd is not None:
                    from src.metrics.tcd import compute_func_cont_tcd
                    fc = compute_func_cont_tcd(cf.tcd, cf.query_tokens)
                    row["func_tcd"] = fc["func_tcd"]
                    row["cont_tcd"] = fc["cont_tcd"]
                    row["tcd_ratio"] = fc["tcd_ratio"]

                model_results.append(row)
                all_results.append(row)

        # Per-model summary
        ss_vals = [r["ss"] for r in model_results]
        print(f"  Mean SS: {np.mean(ss_vals):.4f} ± {np.std(ss_vals):.4f}")
        print(f"  n_tests: {len(model_results)}")

    # Save full results
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "multi_model_results.csv", index=False)

    # Summary table
    summary = (
        df.groupby("model")
        .agg(
            mean_ss=("ss", "mean"),
            std_ss=("ss", "std"),
            mean_gap=("score_gap", "mean"),
            n_tests=("ss", "count"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "multi_model_summary.csv", index=False)

    print("\n=== Multi-Model Comparison Summary ===")
    print(summary.to_string(index=False))

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 7b: Multi-Model Comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "07_multi_model",
    )
    parser.add_argument(
        "--professions-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "professions" / "professions.json",
    )
    parser.add_argument(
        "--n-professions",
        type=int,
        default=33,
    )
    parser.add_argument(
        "--skip-models",
        nargs="*",
        default=[],
        help="Model names to skip (e.g., SPLADE if not installed)",
    )
    args = parser.parse_args()

    queries, name_pairs, template = load_test_config(
        args.professions_json, args.n_professions
    )

    run_model_comparison(
        queries, name_pairs, template,
        args.output_dir,
        skip_models=args.skip_models,
    )


if __name__ == "__main__":
    main()
