"""Backwards-compatibility shim for :mod:`pinecone.inference.models`.

Re-exports the inference result classes that used to live here, plus
:class:`~pinecone.inference.models.index_embed.IndexEmbed`, which is still
defined in this package. Preserved to keep pre-rewrite callers working. New
code should import these names from the top-level :mod:`pinecone` namespace.

:meta private:
"""

from __future__ import annotations

from pinecone.inference.models.index_embed import IndexEmbed
from pinecone.models.inference.embed import EmbeddingsList
from pinecone.models.inference.model_list import ModelInfoList
from pinecone.models.inference.models import ModelInfo
from pinecone.models.inference.rerank import RerankResult

__all__ = [
    "EmbeddingsList",
    "IndexEmbed",
    "ModelInfo",
    "ModelInfoList",
    "RerankResult",
]
