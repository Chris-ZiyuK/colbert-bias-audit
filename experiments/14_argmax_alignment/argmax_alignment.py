"""
MaxSim argmax-alignment analysis.

This experiment asks where the TCD signal enters ColBERT's late interaction:
when a name is swapped, do function-word query tokens change their best-matching
document token more often than content words, and do their matches land near the
identity marker?
"""
import argparse
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.core import (  # noqa: E402
    classify_token,
    encode,
    get_tokens,
    load_model,
    maxsim_detail,
)

scaled_main = importlib.import_module("experiments.30_scaled_experiment.scaled_main")
PROFESSIONS = scaled_main.PROFESSIONS
TEMPLATES = scaled_main.TEMPLATES
build_stratified_pairs = scaled_main.build_stratified_pairs
load_expanded_names = scaled_main.load_expanded_names


def clean_token(tok):
    return tok.replace("##", "").lower()


def find_subsequence(tokens, pattern):
    """Return the first start/end span where pattern appears in tokens."""
    if not pattern:
        return None
    clean_tokens = [clean_token(t) for t in tokens]
    clean_pattern = [clean_token(t) for t in pattern]
    for start in range(0, len(clean_tokens) - len(clean_pattern) + 1):
        if clean_tokens[start:start + len(clean_pattern)] == clean_pattern:
            return (start, start + len(clean_pattern) - 1)
    return None


def in_span(idx, span, window=0):
    if span is None:
        return False
    return span[0] - window <= idx <= span[1] + window


def mean(values):
    return float(np.mean(values)) if values else 0.0


def rate(values):
    return float(np.mean(values)) if values else 0.0


def summarise(rows):
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["query_token_type"]].append(row)

    summary = {}
    for typ, vals in by_type.items():
        summary[typ] = {
            "n_tokens": len(vals),
            "mean_tcd": mean([v["abs_tcd"] for v in vals]),
            "argmax_change_rate": rate([v["argmax_changed"] for v in vals]),
            "name_span_hit_rate": rate([v["either_name_hit"] for v in vals]),
            "name_context_hit_rate_w2": rate([v["either_name_context_hit_w2"] for v in vals]),
            "mean_tcd_if_argmax_changed": mean([v["abs_tcd"] for v in vals if v["argmax_changed"]]),
            "mean_tcd_if_argmax_stable": mean([v["abs_tcd"] for v in vals if not v["argmax_changed"]]),
        }

    func = summary.get("function", {})
    cont = summary.get("content", {})
    if func and cont:
        summary["ratios"] = {
            "tcd_func_over_content": func["mean_tcd"] / cont["mean_tcd"],
            "argmax_change_func_over_content": (
                func["argmax_change_rate"] / cont["argmax_change_rate"]
                if cont["argmax_change_rate"] > 0 else float("inf")
            ),
            "name_context_func_over_content": (
                func["name_context_hit_rate_w2"] / cont["name_context_hit_rate_w2"]
                if cont["name_context_hit_rate_w2"] > 0 else float("inf")
            ),
        }

    transition_counts = Counter(
        (r["doc_token_type_a"], r["doc_token_type_b"])
        for r in rows
        if r["argmax_changed"]
    )
    summary["top_argmax_type_transitions"] = [
        {"from": a, "to": b, "count": c}
        for (a, b), c in transition_counts.most_common(10)
    ]
    return summary


def run(output_dir, max_pairs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model()
    names = load_expanded_names(tokenizer)
    pairs = build_stratified_pairs(names, target_n=120)[:max_pairs]

    rows = []
    total = len(PROFESSIONS) * len(TEMPLATES) * len(pairs)
    completed = 0

    for prof in PROFESSIONS:
        query = f"Who is an experienced {prof}?"
        q_emb = encode(query, tokenizer, model, device, is_query=True)
        q_tokens = get_tokens(query, tokenizer, is_query=True)

        for template_id, template in enumerate(TEMPLATES):
            for pair in pairs:
                name_a = pair["name_a"]["name"]
                name_b = pair["name_b"]["name"]
                doc_a = template.replace("{NAME}", name_a)
                doc_b = template.replace("{NAME}", name_b)

                d_emb_a = encode(doc_a, tokenizer, model, device, is_query=False)
                d_emb_b = encode(doc_b, tokenizer, model, device, is_query=False)
                d_tokens_a = get_tokens(doc_a, tokenizer, is_query=False)
                d_tokens_b = get_tokens(doc_b, tokenizer, is_query=False)

                _, _, scores_a, argmax_a = maxsim_detail(q_emb, d_emb_a)
                _, _, scores_b, argmax_b = maxsim_detail(q_emb, d_emb_b)

                name_span_a = find_subsequence(
                    d_tokens_a,
                    tokenizer.convert_ids_to_tokens(
                        tokenizer.encode(name_a, add_special_tokens=False)
                    ),
                )
                name_span_b = find_subsequence(
                    d_tokens_b,
                    tokenizer.convert_ids_to_tokens(
                        tokenizer.encode(name_b, add_special_tokens=False)
                    ),
                )

                for i, q_tok in enumerate(q_tokens):
                    q_type = classify_token(q_tok)
                    if q_type == "special":
                        continue

                    idx_a = int(argmax_a[i])
                    idx_b = int(argmax_b[i])
                    doc_tok_a = d_tokens_a[idx_a]
                    doc_tok_b = d_tokens_b[idx_b]

                    rows.append({
                        "profession": prof,
                        "template_id": template_id,
                        "name_a": name_a,
                        "name_b": name_b,
                        "pair_type": pair["type"],
                        "query_token": q_tok,
                        "query_token_type": q_type,
                        "doc_token_a": doc_tok_a,
                        "doc_token_b": doc_tok_b,
                        "doc_token_type_a": classify_token(doc_tok_a),
                        "doc_token_type_b": classify_token(doc_tok_b),
                        "argmax_changed": idx_a != idx_b,
                        "abs_tcd": float(abs(scores_a[i] - scores_b[i])),
                        "signed_tcd": float(scores_a[i] - scores_b[i]),
                        "name_hit_a": in_span(idx_a, name_span_a),
                        "name_hit_b": in_span(idx_b, name_span_b),
                        "either_name_hit": in_span(idx_a, name_span_a) or in_span(idx_b, name_span_b),
                        "name_context_hit_a_w2": in_span(idx_a, name_span_a, window=2),
                        "name_context_hit_b_w2": in_span(idx_b, name_span_b, window=2),
                        "either_name_context_hit_w2": (
                            in_span(idx_a, name_span_a, window=2)
                            or in_span(idx_b, name_span_b, window=2)
                        ),
                    })

                completed += 1
                if completed % 250 == 0:
                    print(f"  {completed:,}/{total:,} document pairs processed")

    summary = summarise(rows)
    summary["config"] = {
        "n_name_pairs": len(pairs),
        "n_professions": len(PROFESSIONS),
        "n_templates": len(TEMPLATES),
        "n_document_pairs": total,
        "n_query_token_rows": len(rows),
    }

    with (output_dir / "argmax_alignment_rows.json").open("w") as f:
        json.dump(rows, f)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/14_argmax_alignment")
    parser.add_argument("--max-pairs", type=int, default=24)
    args = parser.parse_args()
    run(args.output_dir, args.max_pairs)
