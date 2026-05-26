"""
Abstract base class for retrieval model wrappers.

All retriever wrappers implement a common interface so that the same
counterfactual auditing pipeline works across ColBERT, DPR, SPLADE, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import numpy as np


@dataclass
class ScoreResult:
    """Result of scoring a single query–document pair.

    Attributes:
        total_score: Scalar relevance score.
        per_token_scores: Optional per-query-token contributions (ColBERT only).
        query_tokens: Optional list of query token strings.
        doc_tokens: Optional list of document token strings.
    """
    total_score: float
    per_token_scores: Optional[np.ndarray] = None
    query_tokens: Optional[List[str]] = None
    doc_tokens: Optional[List[str]] = None


@dataclass
class CounterfactualResult:
    """Result of a counterfactual comparison between two documents.

    Attributes:
        score_a: Total score for document A.
        score_b: Total score for document B.
        score_sensitivity: Normalized score gap |S_A - S_B| / mean(S_A, S_B).
        tcd: Per-token TCD array (ColBERT only).
        query_tokens: Token strings for the query.
        supports_token_level: Whether this model supports per-token decomposition.
    """
    score_a: float
    score_b: float
    score_sensitivity: float
    tcd: Optional[np.ndarray] = None
    query_tokens: Optional[List[str]] = None
    supports_token_level: bool = False


def get_device() -> str:
    """Auto-detect best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BaseRetriever(ABC):
    """Abstract retriever interface for bias auditing.

    Subclasses must implement:
      - load()          → load model weights + tokenizer
      - score(q, d)     → ScoreResult for a single pair
      - model_name      → human-readable model identifier

    The base class provides:
      - counterfactual_score(q, d_a, d_b)  → CounterfactualResult
      - batch_counterfactual(...)          → list of CounterfactualResult
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self._loaded = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier, e.g. 'ColBERTv2'."""
        ...

    @property
    def supports_token_level(self) -> bool:
        """Whether this model supports per-token score decomposition."""
        return False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and tokenizer onto self.device."""
        ...

    @abstractmethod
    def score(self, query: str, document: str) -> ScoreResult:
        """Score a single query–document pair."""
        ...

    def ensure_loaded(self) -> None:
        """Lazy-load model on first use."""
        if not self._loaded:
            self.load()
            self._loaded = True

    def counterfactual_score(
        self, query: str, doc_a: str, doc_b: str
    ) -> CounterfactualResult:
        """Score a counterfactual pair and compute disparity metrics.

        Args:
            query: The professional query, e.g. "Who is a qualified doctor?"
            doc_a: Document with identity marker A.
            doc_b: Document with identity marker B (counterfactual).

        Returns:
            CounterfactualResult with score gap, sensitivity, and optional TCD.
        """
        self.ensure_loaded()

        result_a = self.score(query, doc_a)
        result_b = self.score(query, doc_b)

        s_a = result_a.total_score
        s_b = result_b.total_score
        mean_score = 0.5 * (s_a + s_b)
        ss = abs(s_a - s_b) / mean_score if mean_score > 0 else 0.0

        tcd = None
        if (result_a.per_token_scores is not None
                and result_b.per_token_scores is not None):
            tcd = result_a.per_token_scores - result_b.per_token_scores

        return CounterfactualResult(
            score_a=s_a,
            score_b=s_b,
            score_sensitivity=ss,
            tcd=tcd,
            query_tokens=result_a.query_tokens,
            supports_token_level=self.supports_token_level,
        )

    def batch_counterfactual(
        self,
        query: str,
        doc_pairs: List[tuple],
    ) -> List[CounterfactualResult]:
        """Score multiple counterfactual pairs for the same query.

        Args:
            query: The professional query.
            doc_pairs: List of (doc_a, doc_b) tuples.

        Returns:
            List of CounterfactualResult objects.
        """
        return [
            self.counterfactual_score(query, d_a, d_b)
            for d_a, d_b in doc_pairs
        ]
