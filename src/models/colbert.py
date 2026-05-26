"""
ColBERTv2 retriever wrapper with per-token MaxSim decomposition.

This is the primary model for TCD-based bias auditing, because ColBERT's
late-interaction architecture preserves per-token representations and scores
via MaxSim, enabling token-level attribution of score differences.

Model: colbert-ir/colbertv2.0  (HuggingFace)
Architecture: Late-interaction with per-token L2-normalised BERT embeddings
Scoring: MaxSim — sum of per-query-token maximum cosine similarities
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.models.base import BaseRetriever, ScoreResult


# Default model identifier on HuggingFace
DEFAULT_MODEL_NAME = "colbert-ir/colbertv2.0"

# Maximum sequence length for ColBERTv2
MAX_LENGTH = 128


class ColBERTRetriever(BaseRetriever):
    """ColBERTv2 wrapper supporting per-token MaxSim decomposition.

    This wrapper exposes three levels of detail:
      1. score(q, d) → ScoreResult with total + per-token scores
      2. encode(text, is_query) → L2-normalised token embeddings
      3. maxsim_detail(q_emb, d_emb) → full similarity matrix + argmax indices

    The per-token decomposition is what makes ColBERT uniquely suited for
    Token Contribution Disparity (TCD) analysis.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        max_length: int = MAX_LENGTH,
    ):
        super().__init__(device=device)
        self.model_id = model_id
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

    @property
    def model_name(self) -> str:
        return "ColBERTv2"

    @property
    def model_hf_id(self) -> str:
        """HuggingFace model identifier for loading variants."""
        return self.model_id

    @property
    def supports_token_level(self) -> bool:
        return True

    def load(self) -> None:
        """Load ColBERTv2 model and tokenizer."""
        print(f"Loading {self.model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"  ✓ {self.model_name} loaded ({self.device})")

    def encode(
        self, text: str, is_query: bool = False
    ) -> torch.Tensor:
        """Encode text into L2-normalised ColBERT token embeddings.

        Args:
            text: Raw text to encode.
            is_query: If True, prepend "query: " prefix; else "document: ".

        Returns:
            Tensor of shape (num_tokens, hidden_dim), L2-normalised.
        """
        self.ensure_loaded()
        prefix = "query: " if is_query else "document: "
        inputs = self.tokenizer(
            prefix + text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.model(**inputs).last_hidden_state.squeeze(0)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def get_tokens(self, text: str, is_query: bool = False) -> List[str]:
        """Return token strings for a text (with prefix)."""
        self.ensure_loaded()
        prefix = "query: " if is_query else "document: "
        ids = self.tokenizer.encode(prefix + text)
        return self.tokenizer.convert_ids_to_tokens(ids)

    def get_token_count(self, text: str) -> int:
        """Count BPE subword tokens for a string (no special tokens)."""
        self.ensure_loaded()
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def maxsim_detail(
        q_emb: torch.Tensor, d_emb: torch.Tensor
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Compute MaxSim with full per-token breakdown.

        Args:
            q_emb: Query embeddings, shape (n_q, dim).
            d_emb: Document embeddings, shape (n_d, dim).

        Returns:
            total_score: Sum of per-token max similarities.
            sim_matrix: Full (n_q, n_d) similarity matrix.
            per_token_scores: (n_q,) array of max similarities per query token.
            argmax_indices: (n_q,) array of best-matching doc token indices.
        """
        sim_matrix = torch.matmul(q_emb, d_emb.T)
        per_token, argmax_idx = sim_matrix.max(dim=1)
        return (
            per_token.sum().item(),
            sim_matrix.cpu().numpy(),
            per_token.cpu().numpy(),
            argmax_idx.cpu().numpy(),
        )

    def score(self, query: str, document: str) -> ScoreResult:
        """Score a query–document pair with per-token decomposition.

        Returns ScoreResult containing:
          - total_score: ColBERT MaxSim score
          - per_token_scores: contribution of each query token
          - query_tokens: list of query token strings
          - doc_tokens: list of document token strings
        """
        self.ensure_loaded()

        q_emb = self.encode(query, is_query=True)
        d_emb = self.encode(document, is_query=False)
        q_tokens = self.get_tokens(query, is_query=True)
        d_tokens = self.get_tokens(document, is_query=False)

        total, _, per_token, _ = self.maxsim_detail(q_emb, d_emb)

        return ScoreResult(
            total_score=total,
            per_token_scores=per_token,
            query_tokens=q_tokens,
            doc_tokens=d_tokens,
        )

    def encode_and_detail(
        self, query: str, document: str
    ) -> dict:
        """Full diagnostic output: embeddings + similarity matrix + tokens.

        Useful for attention analysis and visualization experiments.
        """
        self.ensure_loaded()

        q_emb = self.encode(query, is_query=True)
        d_emb = self.encode(document, is_query=False)
        q_tokens = self.get_tokens(query, is_query=True)
        d_tokens = self.get_tokens(document, is_query=False)

        total, sim_matrix, per_token, argmax = self.maxsim_detail(q_emb, d_emb)

        return {
            "q_emb": q_emb,
            "d_emb": d_emb,
            "q_tokens": q_tokens,
            "d_tokens": d_tokens,
            "total_score": total,
            "sim_matrix": sim_matrix,
            "per_token_scores": per_token,
            "argmax_indices": argmax,
        }
