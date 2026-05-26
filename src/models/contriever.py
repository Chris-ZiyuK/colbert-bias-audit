"""
Contriever wrapper for contrastive dense retrieval comparison.

Contriever is a modern unsupervised dense retriever trained with
contrastive learning. Like DPR, it produces single-vector representations
(mean-pooled), so only aggregate score comparison is possible.

Model: facebook/contriever
"""

from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

from src.models.base import BaseRetriever, ScoreResult


DEFAULT_MODEL = "facebook/contriever"


class ContrieverRetriever(BaseRetriever):
    """Contriever wrapper — contrastive dense retriever, aggregate only."""

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
        return "Contriever"

    @property
    def supports_token_level(self) -> bool:
        return False

    def load(self) -> None:
        print(f"Loading Contriever on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"  ✓ Contriever loaded ({self.device})")

    def _mean_pool(
        self, outputs, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling over token embeddings (Contriever style)."""
        token_emb = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        sum_emb = torch.sum(token_emb * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask

    def _encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._mean_pool(outputs, inputs["attention_mask"]).squeeze(0)

    def score(self, query: str, document: str) -> ScoreResult:
        self.ensure_loaded()
        q_emb = self._encode(query)
        d_emb = self._encode(document)
        dot = torch.dot(q_emb, d_emb).item()
        return ScoreResult(total_score=dot)
