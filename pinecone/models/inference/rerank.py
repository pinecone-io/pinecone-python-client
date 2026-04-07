"""Rerank response models for the Inference API."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class RerankUsage(Struct, kw_only=True):
    """Usage information for a rerank request.

    Attributes:
        rerank_units: Number of rerank units consumed.
    """

    rerank_units: int

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. usage['rerank_units'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'rerank_units' in usage``)."""
        return key in self.__struct_fields__


class RankedDocument(Struct, kw_only=True):
    """A document with its relevance score from a rerank operation.

    Attributes:
        index: The original index of the document in the input list.
        score: The relevance score assigned by the reranker.
        document: The original document content, if requested.
    """

    index: int
    score: float
    document: dict[str, Any] | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. doc['score'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'score' in doc``)."""
        return key in self.__struct_fields__


class RerankResult(Struct, kw_only=True):
    """Response from the rerank endpoint.

    Attributes:
        model: The model used for reranking.
        data: The list of ranked documents, ordered by relevance.
        usage: Rerank usage information.
    """

    model: str
    data: list[RankedDocument]
    usage: RerankUsage

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. result['model'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'model' in result``)."""
        return key in self.__struct_fields__
