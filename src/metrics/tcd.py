"""
Token Contribution Disparity (TCD) — core bias attribution metric.

TCD measures how much a single query token's contribution to the ColBERT
MaxSim score changes when the document's identity marker is swapped.

Definitions:
  c_i^A = max_j q_i^T d_j^A    (contribution of q_i when scoring document A)
  TCD(q_i) = |c_i^A - c_i^B|   (absolute per-token disparity)

  Func-TCD = mean TCD over function-word query tokens
  Cont-TCD = mean TCD over content-word query tokens

  Score Sensitivity (SS) = |S_A - S_B| / mean(S_A, S_B)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Token classification
# ============================================================================

FUNCTION_WORDS = {
    # Determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over",
    # Auxiliaries / modals
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "has", "have", "had",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # Pronouns
    "who", "what", "which", "whom", "whose",
    "he", "she", "it", "they", "we", "i", "you",
    "him", "her", "them", "us", "me",
    "his", "its", "their", "our", "my", "your",
    # Conjunctions
    "and", "or", "but", "not", "no", "nor", "so", "yet",
    # Relative / subordinating
    "that", "if", "when", "where", "while", "because", "although",
    # Punctuation (often tokenised as separate tokens)
    "?", ".", ",", "!", ":", ";",
}

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "query", "document", ":", "."}


def classify_token(token: str) -> str:
    """Classify a token as 'function', 'content', or 'special'.

    Args:
        token: A WordPiece/BPE token string (e.g. "is", "##tion", "[CLS]").

    Returns:
        One of 'function', 'content', or 'special'.
    """
    if token in SPECIAL_TOKENS:
        return "special"
    clean = token.replace("##", "").lower()
    if clean in FUNCTION_WORDS:
        return "function"
    return "content"


# ============================================================================
# Core TCD computation
# ============================================================================

def compute_tcd(
    per_token_a: np.ndarray,
    per_token_b: np.ndarray,
) -> np.ndarray:
    """Compute signed TCD for each query token.

    Args:
        per_token_a: Per-token MaxSim contributions for document A, shape (n_q,).
        per_token_b: Per-token MaxSim contributions for document B, shape (n_q,).

    Returns:
        Signed TCD array of shape (n_q,): positive means token favoured doc A.
    """
    return per_token_a - per_token_b


def compute_func_cont_tcd(
    tcd_array: np.ndarray,
    query_tokens: List[str],
    signed: bool = False,
) -> Dict[str, float]:
    """Decompose TCD into function-word and content-word components.

    Args:
        tcd_array: Signed TCD values, shape (n_q,).
        query_tokens: List of query token strings.
        signed: If False (default), uses |TCD|; if True, uses signed TCD.

    Returns:
        Dict with keys: func_tcd, cont_tcd, tcd_ratio, n_func, n_cont,
        func_tokens, cont_tokens.
    """
    func_tcds = []
    cont_tcds = []
    func_tokens = []
    cont_tokens = []

    for i, tok in enumerate(query_tokens):
        cls = classify_token(tok)
        val = tcd_array[i] if signed else abs(tcd_array[i])
        if cls == "function":
            func_tcds.append(val)
            func_tokens.append(tok)
        elif cls == "content":
            cont_tcds.append(val)
            cont_tokens.append(tok)
        # skip 'special'

    func_mean = float(np.mean(func_tcds)) if func_tcds else 0.0
    cont_mean = float(np.mean(cont_tcds)) if cont_tcds else 0.0
    ratio = func_mean / cont_mean if cont_mean > 0 else float("inf")

    return {
        "func_tcd": func_mean,
        "cont_tcd": cont_mean,
        "tcd_ratio": ratio,
        "n_func": len(func_tcds),
        "n_cont": len(cont_tcds),
        "func_tokens": func_tokens,
        "cont_tokens": cont_tokens,
        "func_tcds": func_tcds,
        "cont_tcds": cont_tcds,
    }


def compute_score_sensitivity(score_a: float, score_b: float) -> float:
    """Score Sensitivity: normalised score gap.

    SS = |S_A - S_B| / mean(S_A, S_B)
    """
    mean_score = 0.5 * (score_a + score_b)
    if mean_score <= 0:
        return 0.0
    return abs(score_a - score_b) / mean_score


def compute_full_breakdown(
    per_token_a: np.ndarray,
    per_token_b: np.ndarray,
    query_tokens: List[str],
) -> Dict:
    """Full TCD breakdown: scores, sensitivity, func/cont decomposition.

    Returns a dict suitable for one row in a results DataFrame.
    """
    score_a = float(per_token_a.sum())
    score_b = float(per_token_b.sum())
    ss = compute_score_sensitivity(score_a, score_b)

    tcd = compute_tcd(per_token_a, per_token_b)
    fc = compute_func_cont_tcd(tcd, query_tokens, signed=False)
    fc_signed = compute_func_cont_tcd(tcd, query_tokens, signed=True)

    return {
        "score_a": score_a,
        "score_b": score_b,
        "ss": ss,
        "func_tcd": fc["func_tcd"],
        "cont_tcd": fc["cont_tcd"],
        "tcd_ratio": fc["tcd_ratio"],
        "func_tcd_signed": fc_signed["func_tcd"],
        "cont_tcd_signed": fc_signed["cont_tcd"],
        "n_func": fc["n_func"],
        "n_cont": fc["n_cont"],
        "raw_tcd": tcd,
    }
