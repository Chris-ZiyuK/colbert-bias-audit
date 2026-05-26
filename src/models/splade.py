"""
SPLADE wrapper for sparse neural retrieval comparison.

SPLADE produces sparse bag-of-words representations where each vocabulary
term receives a learned importance weight. This provides partial
decomposability: we can see which *vocabulary terms* contribute to the
score, though not at the same sub-word granularity as ColBERT MaxSim.

Model: naver/splade-cocondenser-ensembledistil
"""

from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.models.base import BaseRetriever, ScoreResult


DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


class SPLADERetriever(BaseRetriever):
    """SPLADE wrapper — sparse neural retriever with per-term weights."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    @property
    def model_name(self) -> str:
        return "SPLADE"

    @property
    def supports_token_level(self) -> bool:
        # SPLADE gives per-vocabulary-term weights (partial decomposition)
        return False

    def load(self) -> None:
        print(f"Loading SPLADE on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"  ✓ SPLADE loaded ({self.device})")

    def _encode_sparse(self, text: str) -> torch.Tensor:
        """Produce SPLADE sparse representation (vocab-sized vector)."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, seq_len, vocab)
        # SPLADE aggregation: max-pool over sequence, then ReLU + log(1+x)
        weights = torch.max(
            torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1),
            dim=1,
        ).values.squeeze(0)
        return weights

    def score(self, query: str, document: str) -> ScoreResult:
        self.ensure_loaded()
        q_sparse = self._encode_sparse(query)
        d_sparse = self._encode_sparse(document)
        dot = torch.dot(q_sparse, d_sparse).item()
        return ScoreResult(total_score=dot)
