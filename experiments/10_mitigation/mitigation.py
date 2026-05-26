"""
Experiment 10: Mitigation Strategies — Tokenizer Equalization and
Function-Word Null-Space Projection.

Two mitigation approaches:

1. Tokenizer Equalization:
   Add top-N rare names to the WordPiece vocabulary so they tokenize as
   single units rather than multi-token fragments. This directly targets
   the rarity amplification effect (1.84×).

2. Null-Space Projection:
   Use Iterative Null-Space Projection (INLP; Ravfogel et al. 2020) to
   identify and remove the identity-encoding subspace from function-word
   representations at inference time.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.colbert import ColBERTRetriever
from src.metrics.tcd import (
    compute_tcd,
    compute_func_cont_tcd,
    compute_score_sensitivity,
)
from src.audit.names import Name


# ==========================================================================
# Mitigation 1: Tokenizer Equalization
# ==========================================================================

def run_tokenizer_equalization(
    model: ColBERTRetriever,
    rare_names: List[str],
    queries: List[str],
    name_pairs: List[Tuple[str, str]],
    template: str,
    output_dir: Path,
) -> Dict:
    """Test whether adding rare names to vocabulary reduces rarity amplification.

    Steps:
      1. Measure baseline TCD with original tokenizer
      2. Add rare names to tokenizer vocabulary
      3. Re-measure TCD with augmented tokenizer
      4. Compare rarity amplification ratios
    """
    results_before = []
    results_after = []

    # === Phase 1: Baseline measurement ===
    print("Phase 1: Baseline measurement...")
    for query in tqdm(queries, desc="Baseline"):
        for name_a, name_b in name_pairs:
            doc_a = template.replace("{NAME}", name_a)
            doc_b = template.replace("{NAME}", name_b)

            # Token counts
            bpe_a = model.get_token_count(name_a)
            bpe_b = model.get_token_count(name_b)

            result_a = model.score(query, doc_a)
            result_b = model.score(query, doc_b)

            tcd = compute_tcd(result_a.per_token_scores, result_b.per_token_scores)
            fc = compute_func_cont_tcd(tcd, result_a.query_tokens)
            ss = compute_score_sensitivity(result_a.total_score, result_b.total_score)

            results_before.append({
                "query": query,
                "name_a": name_a,
                "name_b": name_b,
                "bpe_a": bpe_a,
                "bpe_b": bpe_b,
                "func_tcd": fc["func_tcd"],
                "ss": ss,
                "phase": "before",
            })

    # === Phase 2: Augment tokenizer ===
    print(f"\nPhase 2: Adding {len(rare_names)} names to vocabulary...")
    # Add tokens to tokenizer
    num_added = model.tokenizer.add_tokens(rare_names)
    print(f"  Added {num_added} new tokens to vocabulary")

    # Resize model embeddings
    model.model.resize_token_embeddings(len(model.tokenizer))

    # Verify tokenization changed
    for name in rare_names[:3]:
        tokens_after = model.tokenizer.tokenize(name)
        print(f"  {name}: {tokens_after}")

    # === Phase 3: Post-equalization measurement ===
    print("\nPhase 3: Post-equalization measurement...")
    for query in tqdm(queries, desc="Post-equalization"):
        for name_a, name_b in name_pairs:
            doc_a = template.replace("{NAME}", name_a)
            doc_b = template.replace("{NAME}", name_b)

            bpe_a = model.get_token_count(name_a)
            bpe_b = model.get_token_count(name_b)

            result_a = model.score(query, doc_a)
            result_b = model.score(query, doc_b)

            tcd = compute_tcd(result_a.per_token_scores, result_b.per_token_scores)
            fc = compute_func_cont_tcd(tcd, result_a.query_tokens)
            ss = compute_score_sensitivity(result_a.total_score, result_b.total_score)

            results_after.append({
                "query": query,
                "name_a": name_a,
                "name_b": name_b,
                "bpe_a": bpe_a,
                "bpe_b": bpe_b,
                "func_tcd": fc["func_tcd"],
                "ss": ss,
                "phase": "after",
            })

    # === Save and summarize ===
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = results_before + results_after
    with open(output_dir / "tokenizer_equalization_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    before_ss = np.mean([r["ss"] for r in results_before])
    after_ss = np.mean([r["ss"] for r in results_after])
    before_ftcd = np.mean([r["func_tcd"] for r in results_before])
    after_ftcd = np.mean([r["func_tcd"] for r in results_after])

    summary = {
        "before_mean_ss": float(before_ss),
        "after_mean_ss": float(after_ss),
        "ss_reduction_pct": float((before_ss - after_ss) / before_ss * 100),
        "before_mean_func_tcd": float(before_ftcd),
        "after_mean_func_tcd": float(after_ftcd),
        "func_tcd_reduction_pct": float(
            (before_ftcd - after_ftcd) / before_ftcd * 100
            if before_ftcd > 0 else 0
        ),
        "n_names_added": num_added,
    }

    print("\n=== Tokenizer Equalization Results ===")
    print(f"  SS: {before_ss:.4f} → {after_ss:.4f} "
          f"({summary['ss_reduction_pct']:.1f}% reduction)")
    print(f"  Func-TCD: {before_ftcd:.4f} → {after_ftcd:.4f} "
          f"({summary['func_tcd_reduction_pct']:.1f}% reduction)")

    with open(output_dir / "tokenizer_equalization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ==========================================================================
# Mitigation 2: Null-Space Projection (INLP)
# ==========================================================================

def run_nullspace_projection(
    model: ColBERTRetriever,
    queries: List[str],
    name_pairs: List[Tuple[str, str]],
    template: str,
    output_dir: Path,
    n_projections: int = 5,
) -> Dict:
    """Apply INLP to remove identity information from function-word embeddings.

    Steps:
      1. Collect function-word embeddings across identity groups
      2. Train linear classifier to predict identity from function-word reps
      3. Project out the discriminating direction
      4. Repeat n_projections times (iterative)
      5. Measure TCD with projected embeddings
    """
    from sklearn.linear_model import LogisticRegression

    print("Phase 1: Collecting function-word embeddings...")
    # Collect embeddings
    func_embeddings = []  # [(embedding, identity_label)]

    for query in queries[:5]:
        for name_a, name_b in name_pairs:
            for name, label in [(name_a, 0), (name_b, 1)]:
                doc = template.replace("{NAME}", name)
                emb = model.encode(doc, is_query=False).cpu().numpy()
                tokens = model.get_tokens(doc, is_query=False)

                from src.metrics.tcd import classify_token
                for i, tok in enumerate(tokens):
                    if classify_token(tok) == "function":
                        func_embeddings.append((emb[i], label))

    if not func_embeddings:
        print("No function-word embeddings collected.")
        return {}

    X = np.array([e[0] for e in func_embeddings])
    y = np.array([e[1] for e in func_embeddings])

    print(f"  Collected {len(X)} function-word embeddings")

    # INLP: iteratively find and project out discriminating directions
    projection_matrix = np.eye(X.shape[1])

    for iteration in range(n_projections):
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X, y)
        accuracy = clf.score(X, y)
        print(f"  Iteration {iteration + 1}: classifier accuracy = {accuracy:.3f}")

        if accuracy < 0.55:
            print("  Classifier near chance — stopping INLP.")
            break

        # Get discriminating direction
        w = clf.coef_[0]
        w = w / np.linalg.norm(w)

        # Project out this direction
        P = np.eye(X.shape[1]) - np.outer(w, w)
        projection_matrix = P @ projection_matrix
        X = X @ P.T

    print(f"\nPhase 2: Measuring TCD with projected embeddings...")

    # Now measure TCD with projection applied to function-word embeddings
    results_before = []
    results_after = []

    for query in tqdm(queries[:10], desc="Null-space projection"):
        for name_a, name_b in name_pairs[:5]:
            doc_a = template.replace("{NAME}", name_a)
            doc_b = template.replace("{NAME}", name_b)

            # Original
            result_a = model.score(query, doc_a)
            result_b = model.score(query, doc_b)
            tcd_orig = compute_tcd(result_a.per_token_scores, result_b.per_token_scores)
            fc_orig = compute_func_cont_tcd(tcd_orig, result_a.query_tokens)

            results_before.append({
                "func_tcd": fc_orig["func_tcd"],
                "cont_tcd": fc_orig["cont_tcd"],
                "ratio": fc_orig["tcd_ratio"],
            })

            # With projection: project function-word embeddings
            q_emb = model.encode(query, is_query=True)
            d_emb_a = model.encode(doc_a, is_query=False)
            d_emb_b = model.encode(doc_b, is_query=False)
            tokens_d = model.get_tokens(doc_a, is_query=False)

            P_tensor = torch.tensor(
                projection_matrix, dtype=d_emb_a.dtype, device=d_emb_a.device
            )

            # Project only function-word positions
            d_emb_a_proj = d_emb_a.clone()
            d_emb_b_proj = d_emb_b.clone()
            for i, tok in enumerate(tokens_d):
                if classify_token(tok) == "function" and i < d_emb_a_proj.shape[0]:
                    d_emb_a_proj[i] = d_emb_a_proj[i] @ P_tensor.T
                    d_emb_b_proj[i] = d_emb_b_proj[i] @ P_tensor.T

            # Re-normalise
            d_emb_a_proj = torch.nn.functional.normalize(d_emb_a_proj, p=2, dim=-1)
            d_emb_b_proj = torch.nn.functional.normalize(d_emb_b_proj, p=2, dim=-1)

            # Re-score
            _, _, pts_a, _ = ColBERTRetriever.maxsim_detail(q_emb, d_emb_a_proj)
            _, _, pts_b, _ = ColBERTRetriever.maxsim_detail(q_emb, d_emb_b_proj)

            q_tokens = model.get_tokens(query, is_query=True)
            tcd_proj = compute_tcd(pts_a, pts_b)
            fc_proj = compute_func_cont_tcd(tcd_proj, q_tokens)

            results_after.append({
                "func_tcd": fc_proj["func_tcd"],
                "cont_tcd": fc_proj["cont_tcd"],
                "ratio": fc_proj["tcd_ratio"],
            })

    output_dir.mkdir(parents=True, exist_ok=True)

    before_ftcd = np.mean([r["func_tcd"] for r in results_before])
    after_ftcd = np.mean([r["func_tcd"] for r in results_after])
    before_ratio = np.mean([r["ratio"] for r in results_before])
    after_ratio = np.mean([r["ratio"] for r in results_after])

    summary = {
        "before_func_tcd": float(before_ftcd),
        "after_func_tcd": float(after_ftcd),
        "func_tcd_reduction_pct": float(
            (before_ftcd - after_ftcd) / before_ftcd * 100
            if before_ftcd > 0 else 0
        ),
        "before_ratio": float(before_ratio),
        "after_ratio": float(after_ratio),
        "n_projections": n_projections,
    }

    print("\n=== Null-Space Projection Results ===")
    print(f"  Func-TCD: {before_ftcd:.4f} → {after_ftcd:.4f}")
    print(f"  Ratio: {before_ratio:.2f}× → {after_ratio:.2f}×")

    with open(output_dir / "nullspace_projection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ==========================================================================
# Main
# ==========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 10: Mitigation Strategies"
    )
    parser.add_argument(
        "--strategy",
        choices=["tokenizer", "nullspace", "all"],
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "10_mitigation",
    )
    parser.add_argument(
        "--professions-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "professions" / "professions.json",
    )
    args = parser.parse_args()

    model = ColBERTRetriever()
    model.load()

    with open(args.professions_json) as f:
        professions = json.load(f)["professions"]
    queries = [p["query"] for p in professions[:10]]

    template = "{NAME} has over ten years of experience in this field."

    name_pairs = [
        ("Emily", "Lakisha"),
        ("Greg", "Jamal"),
        ("Sarah", "Tanisha"),
        ("Todd", "Tyrone"),
        ("Brad", "Darnell"),
    ]

    rare_names = [
        "Lakisha", "Tanisha", "Latoya", "Latonya", "Tamika",
        "Jermaine", "Tremayne", "Rasheed", "Darnell", "Thiruvengadam",
    ]

    strategies = (
        [args.strategy] if args.strategy != "all"
        else ["tokenizer", "nullspace"]
    )

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Running mitigation: {strategy}")
        print(f"{'='*60}")

        if strategy == "tokenizer":
            run_tokenizer_equalization(
                model, rare_names, queries, name_pairs, template,
                args.output_dir / "tokenizer",
            )
        elif strategy == "nullspace":
            run_nullspace_projection(
                model, queries, name_pairs, template,
                args.output_dir / "nullspace",
            )


if __name__ == "__main__":
    main()
