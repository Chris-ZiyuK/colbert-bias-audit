"""
Model wrappers for bias auditing across retrieval architectures.

Supported models:
- ColBERTv2 (late-interaction, token-level decomposition)
- DPR (single-vector dense retriever)
- SPLADE (sparse neural retriever)
- Cross-Encoder (reranker, e.g., MiniLM)
- Contriever (contrastive dense retriever)
"""

from src.models.base import BaseRetriever
from src.models.colbert import ColBERTRetriever

__all__ = ["BaseRetriever", "ColBERTRetriever"]
