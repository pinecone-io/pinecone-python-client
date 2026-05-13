"""Backwards-compatibility shim for :mod:`pinecone.inference`.

Re-exports classes that used to live at :mod:`pinecone.inference` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Backwards-compatibility re-exports (see docs/conventions/backcompat-shims.md)
# :meta private:
# ---------------------------------------------------------------------------
from pinecone.inference.inference import Inference as _Inference
from pinecone.inference.inference_asyncio import AsyncioInference as _AsyncioInference
from pinecone.models.enums import EmbedModel, RerankModel
from pinecone.models.inference.embed import EmbeddingsList
from pinecone.models.inference.model_list import ModelInfoList
from pinecone.models.inference.models import ModelInfo
from pinecone.models.inference.rerank import RerankResult

Inference = _Inference
"""Backwards-compatibility alias. Canonical: :class:`pinecone.client.inference.Inference`.

:meta private:
"""
AsyncioInference = _AsyncioInference
"""Backwards-compatibility alias.

Canonical: :class:`pinecone.async_client.inference.AsyncInference`.

:meta private:
"""

__all__ = [
    "AsyncioInference",
    "EmbedModel",
    "EmbeddingsList",
    "Inference",
    "ModelInfo",
    "ModelInfoList",
    "RerankModel",
    "RerankResult",
]
