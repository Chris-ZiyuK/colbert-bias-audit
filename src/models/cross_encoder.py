"""
Cross-encoder reranker wrapper for bias comparison.

Cross-encoders jointly encode query and document through a single BERT
pass, producing a scalar relevance score. They are the strongest rankers
but have no decomposability at all — purely aggregate scoring.

Model: cross-encoder/ms-marco-MiniLM-L-12-v2
"""

from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.base import BaseRetriever, ScoreResult


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class CrossEncoderRetriever(BaseRetriever):
    """Cross-encoder wrapper — joint encoding, aggregate score only."""

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
        return "CrossEncoder-MiniLM"

    @property
    def supports_token_level(self) -> bool:
        return False

    def load(self) -> None:
        print(f"Loading CrossEncoder on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"  ✓ CrossEncoder loaded ({self.device})")

    def score(self, query: str, document: str) -> ScoreResult:
        self.ensure_loaded()
        inputs = self.tokenizer(
            query,
            document,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
        return ScoreResult(total_score=logits.item())
