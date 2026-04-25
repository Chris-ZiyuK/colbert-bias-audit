"""
Shared utilities for ColBERT bias audit experiments.
=====================================================
Core functions: encode, get_tokens, maxsim_detail, classify_token, cosine_sim.
Extracted from phase1_synthetic experiments for reuse across P1/P2/future phases.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "colbert-ir/colbertv2.0"

# Function words set (expanded)
FUNCTION_WORDS = {
    "who", "what", "which", "is", "are", "was", "were", "a", "an", "the",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "not", "no", "do", "does", "did", "has", "have",
    "had", "can", "could", "will", "would", "shall", "should", "may",
    "might", "must", "be", "been", "being", "that", "this", "these",
    "those", "it", "its", "?", ".", ",", "!", ":", ";",
}
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "query", "document", ":", "."}


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device=None):
    """Load ColBERTv2 model and tokenizer."""
    if device is None:
        device = get_device()
    print(f"🔧 Device: {device}")
    print("📦 Loading ColBERTv2...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("✅ Model loaded!\n")
    return tokenizer, model, device


def encode(text, tokenizer, model, device, is_query=False):
    """Encode text to L2-normalized ColBERT embeddings."""
    prefix = "query: " if is_query else "document: "
    inputs = tokenizer(prefix + text, return_tensors="pt", padding=True,
                       truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.squeeze(0)
    return torch.nn.functional.normalize(emb, p=2, dim=-1)


def get_tokens(text, tokenizer, is_query=False):
    """Get token strings for a text."""
    prefix = "query: " if is_query else "document: "
    ids = tokenizer.encode(prefix + text)
    return tokenizer.convert_ids_to_tokens(ids)


def get_token_count(name, tokenizer):
    """Count BPE tokens for a name (no special tokens)."""
    return len(tokenizer.encode(name, add_special_tokens=False))


def maxsim_detail(q_emb, d_emb):
    """Compute MaxSim with per-token breakdown."""
    M = torch.matmul(q_emb, d_emb.T)
    per_token, argmax_idx = M.max(dim=1)
    return (per_token.sum().item(), M.cpu().numpy(),
            per_token.cpu().numpy(), argmax_idx.cpu().numpy())


def classify_token(tok):
    """Classify a token as 'function', 'content', or 'special'."""
    clean = tok.replace("##", "").lower()
    if tok in SPECIAL_TOKENS:
        return "special"
    if clean in FUNCTION_WORDS:
        return "function"
    return "content"


def cosine_sim(v1, v2):
    """Cosine similarity between two vectors."""
    return torch.nn.functional.cosine_similarity(
        v1.unsqueeze(0), v2.unsqueeze(0)
    ).item()


def compute_tcd_breakdown(q_emb, d_emb_a, d_emb_b, q_tokens):
    """
    Compute TCD breakdown: function-word TCD, content-word TCD, 
    total score sensitivity.
    
    Returns: dict with ss, func_tcd, cont_tcd, raw_tcd array
    """
    s_a, _, scores_a, _ = maxsim_detail(q_emb, d_emb_a)
    s_b, _, scores_b, _ = maxsim_detail(q_emb, d_emb_b)
    
    tcd = scores_a - scores_b
    ss = abs(s_a - s_b) / (0.5 * (s_a + s_b)) if (s_a + s_b) > 0 else 0
    
    func_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens) 
                 if classify_token(t) == "function"]
    cont_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens) 
                 if classify_token(t) == "content"]
    
    func_tcd = np.mean(func_tcds) if func_tcds else 0
    cont_tcd = np.mean(cont_tcds) if cont_tcds else 0
    
    return {
        "score_a": s_a,
        "score_b": s_b,
        "ss": ss,
        "func_tcd": func_tcd,
        "cont_tcd": cont_tcd,
        "tcd_ratio": func_tcd / cont_tcd if cont_tcd > 0 else float('inf'),
        "raw_tcd": tcd,
    }
