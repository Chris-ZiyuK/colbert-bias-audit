"""
DPR (Dense Passage Retrieval) wrapper for aggregate-level bias comparison.

DPR uses separate query and passage encoders to produce single-vector dense
representations. Because it collapses all tokens into one embedding,
per-token TCD decomposition is NOT possible—only aggregate score gaps.

This limitation is precisely why ColBERT's late interaction is valuable
for bias diagnostics: DPR can detect *that* bias exists but not *where*
in the query it originates.

Model: facebook/dpr-question_encoder-single-nq-base
       facebook/dpr-ctx_encoder-single-nq-base
"""

from typing import Optional

import numpy as np
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

from src.models.base import BaseRetriever, ScoreResult


DEFAULT_Q_MODEL = "facebook/dpr-question_encoder-single-nq-base"
DEFAULT_D_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"


class DPRRetriever(BaseRetriever):
    """DPR wrapper — single-vector dense retriever (aggregate-level only).

    Scores documents via dot product of [CLS] embeddings from separate
    query and context encoders. No per-token decomposition.
    """

    def __init__(
        self,
        q_model_id: str = DEFAULT_Q_MODEL,
        d_model_id: str = DEFAULT_D_MODEL,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.q_model_id = q_model_id
        self.d_model_id = d_model_id
        self.q_tokenizer = None
        self.d_tokenizer = None
        self.q_model = None
        self.d_model = None

    @property
    def model_name(self) -> str:
        return "DPR"

    @property
    def supports_token_level(self) -> bool:
        return False

    def load(self) -> None:
        print(f"Loading DPR encoders on {self.device}...")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            self.q_model_id
        )
        self.q_model = DPRQuestionEncoder.from_pretrained(
            self.q_model_id
        ).to(self.device)
        self.q_model.eval()

        self.d_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            self.d_model_id
        )
        self.d_model = DPRContextEncoder.from_pretrained(
            self.d_model_id
        ).to(self.device)
        self.d_model.eval()

        self._loaded = True
        print(f"  ✓ DPR loaded ({self.device})")

    def _encode_query(self, query: str) -> torch.Tensor:
        inputs = self.q_tokenizer(
            query, return_tensors="pt", truncation=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.q_model(**inputs).pooler_output.squeeze(0)
        return emb

    def _encode_doc(self, document: str) -> torch.Tensor:
        inputs = self.d_tokenizer(
            document, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.d_model(**inputs).pooler_output.squeeze(0)
        return emb

    def score(self, query: str, document: str) -> ScoreResult:
        self.ensure_loaded()
        q_emb = self._encode_query(query)
        d_emb = self._encode_doc(document)
        dot = torch.dot(q_emb, d_emb).item()
        return ScoreResult(total_score=dot)
