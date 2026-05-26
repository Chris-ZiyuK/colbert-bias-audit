"""
Experiment 7a: Attention Ablation — Mechanistic validation of function-word
bias propagation.

This experiment tests whether function words are a *causal* pathway for bias
propagation, not merely a correlate. Three sub-experiments:

  1. Name Masking:  Replace name tokens with [MASK] → does function-word TCD
     drop to near zero?
  2. Attention Rollout: Compute attention flow (Abnar & Zuidema 2020) from
     name positions → do function words receive more attention from names
     than content words?
  3. Identity Probing: Train linear probes at each token position to predict
     identity → are function-word positions more predictive?

This is Professor Suresh's #1 recommended follow-up experiment.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.colbert import ColBERTRetriever
from src.metrics.tcd import (
    classify_token,
    compute_tcd,
    compute_func_cont_tcd,
)
from src.metrics.stats import mann_whitney_one_sided, cohens_d


# ==========================================================================
# Sub-experiment 1: Name Masking
# ==========================================================================

def run_name_masking(
    model: ColBERTRetriever,
    queries: List[str],
    doc_template: str,
    name_pairs: List[Tuple[str, str]],
    output_dir: Path,
) -> Dict:
    """Replace name tokens with [MASK] and measure if function-word TCD drops.

    Logic:
      If function-word TCD is caused by name propagation through attention,
      then masking the name should eliminate the signal.

    Steps:
      1. Score (query, doc_with_name_A) and (query, doc_with_name_B) → TCD_original
      2. Score (query, doc_with_[MASK]_A) and (query, doc_with_[MASK]_B) → TCD_masked
      3. Compare Func-TCD between original and masked conditions
    """
    results = []

    for query in tqdm(queries, desc="Name masking"):
        for name_a, name_b in name_pairs:
            # Original documents
            doc_a = doc_template.replace("{NAME}", name_a)
            doc_b = doc_template.replace("{NAME}", name_b)

            result_a = model.score(query, doc_a)
            result_b = model.score(query, doc_b)
            tcd_orig = compute_tcd(
                result_a.per_token_scores, result_b.per_token_scores
            )
            fc_orig = compute_func_cont_tcd(tcd_orig, result_a.query_tokens)

            # Masked documents — replace name with [MASK]
            doc_a_masked = doc_template.replace("{NAME}", "[MASK]")
            doc_b_masked = doc_a_masked  # Both are identical now

            result_a_m = model.score(query, doc_a_masked)
            result_b_m = model.score(query, doc_b_masked)
            tcd_masked = compute_tcd(
                result_a_m.per_token_scores, result_b_m.per_token_scores
            )
            fc_masked = compute_func_cont_tcd(tcd_masked, result_a_m.query_tokens)

            results.append({
                "query": query,
                "name_a": name_a,
                "name_b": name_b,
                "func_tcd_original": fc_orig["func_tcd"],
                "cont_tcd_original": fc_orig["cont_tcd"],
                "ratio_original": fc_orig["tcd_ratio"],
                "func_tcd_masked": fc_masked["func_tcd"],
                "cont_tcd_masked": fc_masked["cont_tcd"],
                "ratio_masked": fc_masked["tcd_ratio"],
                "func_tcd_reduction": (
                    (fc_orig["func_tcd"] - fc_masked["func_tcd"])
                    / fc_orig["func_tcd"]
                    if fc_orig["func_tcd"] > 0
                    else 0.0
                ),
            })

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "name_masking_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary statistics
    func_orig = np.array([r["func_tcd_original"] for r in results])
    func_masked = np.array([r["func_tcd_masked"] for r in results])
    reductions = np.array([r["func_tcd_reduction"] for r in results])

    summary = {
        "n_tests": len(results),
        "mean_func_tcd_original": float(np.mean(func_orig)),
        "mean_func_tcd_masked": float(np.mean(func_masked)),
        "mean_reduction_pct": float(np.mean(reductions) * 100),
        "median_reduction_pct": float(np.median(reductions) * 100),
    }

    print("\n=== Name Masking Results ===")
    print(f"  Func-TCD original:  {summary['mean_func_tcd_original']:.4f}")
    print(f"  Func-TCD masked:    {summary['mean_func_tcd_masked']:.4f}")
    print(f"  Mean reduction:     {summary['mean_reduction_pct']:.1f}%")

    with open(output_dir / "name_masking_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ==========================================================================
# Sub-experiment 2: Attention Rollout
# ==========================================================================

# Module-level eager model for attention rollout (lazy-loaded)
_eager_model = None

def _get_eager_model(model: ColBERTRetriever):
    """Load a copy of the model with eager attention for output_attentions support."""
    global _eager_model
    if _eager_model is None:
        from transformers import AutoModel
        _eager_model = AutoModel.from_pretrained(
            model.model_hf_id, attn_implementation="eager"
        ).to(model.device)
        _eager_model.eval()
        print("  ✓ Loaded eager-attention model for rollout")
    return _eager_model


def compute_attention_rollout(
    model: ColBERTRetriever,
    text: str,
    is_query: bool = False,
) -> np.ndarray:
    """Compute attention rollout matrix (Abnar & Zuidema, 2020).

    Attention rollout tracks how attention flows through all layers
    by taking the product of attention matrices (plus residual connections).

    Returns:
        rollout_matrix: (n_tokens, n_tokens) matrix where entry [i,j] represents
        the total attention flow from token j to token i.
    """
    eager_model = _get_eager_model(model)
    model.ensure_loaded()
    prefix = "query: " if is_query else "document: "
    inputs = model.tokenizer(
        prefix + text,
        return_tensors="pt",
        truncation=True,
        max_length=model.max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = eager_model(**inputs, output_attentions=True)

    # Stack attention: (n_layers, batch, n_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions)
    # Average over heads, squeeze batch: (n_layers, seq_len, seq_len)
    att_avg = attentions[:, 0].mean(dim=1).cpu().numpy()

    n_layers, seq_len, _ = att_avg.shape

    # Rollout: iteratively multiply attention matrices with residual
    rollout = np.eye(seq_len)
    for layer_idx in range(n_layers):
        # Add residual connection: 0.5 * I + 0.5 * Attn
        att_with_residual = 0.5 * np.eye(seq_len) + 0.5 * att_avg[layer_idx]
        # Row-normalise
        att_with_residual = att_with_residual / att_with_residual.sum(
            axis=-1, keepdims=True
        )
        rollout = rollout @ att_with_residual

    return rollout


def run_attention_rollout(
    model: ColBERTRetriever,
    documents: List[str],
    name_positions: Optional[List[int]] = None,
    output_dir: Path = Path("results"),
) -> Dict:
    """Measure attention flow from name tokens to function vs content words.

    For each document, compute how much total attention flows from the
    name token position(s) to function-word positions vs content-word positions.
    """
    results = []

    for doc in tqdm(documents, desc="Attention rollout"):
        rollout = compute_attention_rollout(model, doc, is_query=False)
        tokens = model.get_tokens(doc, is_query=False)

        # Find name position (first non-special, non-prefix token)
        name_idx = None
        for i, tok in enumerate(tokens):
            cls = classify_token(tok)
            if cls == "content" and tok not in ["document", ":"]:
                name_idx = i
                break

        if name_idx is None:
            continue

        # Measure attention from name to each other token
        attention_from_name = rollout[:, name_idx]

        func_attention = []
        cont_attention = []
        for i, tok in enumerate(tokens):
            if i == name_idx:
                continue
            cls = classify_token(tok)
            if cls == "function":
                func_attention.append(attention_from_name[i])
            elif cls == "content":
                cont_attention.append(attention_from_name[i])

        if func_attention and cont_attention:
            results.append({
                "document": doc[:80],
                "name_token": tokens[name_idx],
                "mean_func_attention": float(np.mean(func_attention)),
                "mean_cont_attention": float(np.mean(cont_attention)),
                "attention_ratio": (
                    float(np.mean(func_attention) / np.mean(cont_attention))
                    if np.mean(cont_attention) > 0
                    else float("inf")
                ),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "attention_rollout_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if results:
        ratios = [r["attention_ratio"] for r in results]
        print("\n=== Attention Rollout Results ===")
        print(f"  Mean func/cont attention ratio: {np.mean(ratios):.2f}×")
        print(f"  n_documents: {len(results)}")

    return {"results": results}


# ==========================================================================
# Sub-experiment 3: Identity Probing
# ==========================================================================

def run_identity_probing(
    model: ColBERTRetriever,
    documents_by_identity: Dict[str, List[str]],
    output_dir: Path = Path("results"),
) -> Dict:
    """Train linear probes on per-position representations to predict identity.

    If function-word positions encode more identity information,
    probing accuracy at those positions should be higher.

    Args:
        documents_by_identity: {identity_label: [doc1, doc2, ...]}
        output_dir: Directory to save results.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Collect representations at each token position
    all_embeddings = {}  # position_idx -> list of (embedding, label)
    all_tokens_at_pos = {}

    for label_idx, (identity, docs) in enumerate(
        documents_by_identity.items()
    ):
        for doc in tqdm(docs, desc=f"Encoding {identity}"):
            model.ensure_loaded()
            emb = model.encode(doc, is_query=False).cpu().numpy()
            tokens = model.get_tokens(doc, is_query=False)

            for pos in range(min(len(tokens), emb.shape[0])):
                if pos not in all_embeddings:
                    all_embeddings[pos] = []
                    all_tokens_at_pos[pos] = tokens[pos]
                all_embeddings[pos].append((emb[pos], label_idx))

    # Probe at each position
    probe_results = []
    for pos in sorted(all_embeddings.keys()):
        data = all_embeddings[pos]
        if len(data) < 20:
            continue

        X = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])

        if len(set(y)) < 2:
            continue

        clf = LogisticRegression(max_iter=500, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

        tok = all_tokens_at_pos.get(pos, f"[pos_{pos}]")
        tok_class = classify_token(tok)

        probe_results.append({
            "position": pos,
            "token": tok,
            "token_class": tok_class,
            "probe_accuracy": float(np.mean(scores)),
            "probe_std": float(np.std(scores)),
            "n_samples": len(data),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "probing_results.json", "w") as f:
        json.dump(probe_results, f, indent=2)

    # Summary by token class
    func_acc = [r["probe_accuracy"] for r in probe_results
                if r["token_class"] == "function"]
    cont_acc = [r["probe_accuracy"] for r in probe_results
                if r["token_class"] == "content"]

    if func_acc and cont_acc:
        print("\n=== Identity Probing Results ===")
        print(f"  Function-word probe accuracy: {np.mean(func_acc):.3f}")
        print(f"  Content-word probe accuracy:  {np.mean(cont_acc):.3f}")

        test = mann_whitney_one_sided(
            np.array(func_acc), np.array(cont_acc), "greater"
        )
        print(f"  Mann-Whitney p = {test['p_value']:.4f}")

    return {"probe_results": probe_results}


# ==========================================================================
# Main entry point
# ==========================================================================

def main():
    """Run all attention ablation sub-experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 7a: Attention Ablation"
    )
    parser.add_argument(
        "--sub",
        choices=["masking", "rollout", "probing", "all"],
        default="all",
        help="Which sub-experiment(s) to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "07_attention_ablation",
    )
    parser.add_argument(
        "--professions-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "professions" / "professions.json",
    )
    args = parser.parse_args()

    # Load model
    model = ColBERTRetriever()
    model.load()

    # Load professions
    with open(args.professions_json) as f:
        professions = json.load(f)["professions"]
    queries = [p["query"] for p in professions[:10]]  # Use first 10

    # Name pairs
    name_pairs = [
        ("Emily", "Lakisha"),
        ("Greg", "Jamal"),
        ("Sarah", "Tanisha"),
        ("Todd", "Tyrone"),
        ("Allison", "Latoya"),
    ]

    # Doc template
    template = "{NAME} has over ten years of experience in this field."

    subs = [args.sub] if args.sub != "all" else ["masking", "rollout", "probing"]

    for sub in subs:
        print(f"\n{'='*60}")
        print(f"Running sub-experiment: {sub}")
        print(f"{'='*60}")

        if sub == "masking":
            run_name_masking(
                model, queries, template, name_pairs,
                args.output_dir / "masking",
            )

        elif sub == "rollout":
            docs = [
                template.replace("{NAME}", n)
                for pair in name_pairs
                for n in pair
            ]
            run_attention_rollout(
                model, docs,
                output_dir=args.output_dir / "rollout",
            )

        elif sub == "probing":
            docs_by_identity = {}
            for name_a, name_b in name_pairs:
                docs_by_identity.setdefault("group_a", []).extend([
                    template.replace("{NAME}", name_a),
                    f"{name_a} is a highly qualified professional.",
                    f"{name_a} brings extensive expertise to their work.",
                ])
                docs_by_identity.setdefault("group_b", []).extend([
                    template.replace("{NAME}", name_b),
                    f"{name_b} is a highly qualified professional.",
                    f"{name_b} brings extensive expertise to their work.",
                ])
            run_identity_probing(
                model, docs_by_identity,
                output_dir=args.output_dir / "probing",
            )

    print("\n✓ All attention ablation experiments complete.")


if __name__ == "__main__":
    main()
