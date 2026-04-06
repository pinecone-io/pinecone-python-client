"""Inference namespace — embed and rerank operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.inference_adapter import (
    InferenceAdapter,
    normalize_embed_inputs,
    normalize_rerank_documents,
)
from pinecone._internal.constants import INFERENCE_API_VERSION
from pinecone._internal.validation import require_non_empty
from pinecone.models import enums as _enums

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig
    from pinecone.models.inference.embed import EmbeddingsList
    from pinecone.models.inference.rerank import RerankResult

logger = logging.getLogger(__name__)


class Inference:
    """Control-plane operations for Pinecone inference (embed & rerank).

    Provides methods to generate embeddings and rerank documents using
    Pinecone's hosted models.

    Args:
        config (PineconeConfig): SDK configuration used to construct an
            HTTP client targeting the inference API version.

    Examples:

        from pinecone import Pinecone

        pc = Pinecone(api_key="your-api-key")
        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=["Hello, world!"],
        )
    """

    EmbedModel = _enums.EmbedModel
    RerankModel = _enums.RerankModel

    def __init__(self, config: PineconeConfig) -> None:
        from pinecone._internal.http_client import HTTPClient

        self._http = HTTPClient(config, INFERENCE_API_VERSION)
        self._adapter = InferenceAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Inference()"

    def embed(
        self,
        model: _enums.EmbedModel | str,
        inputs: str | list[str] | list[dict[str, Any]],
        parameters: dict[str, Any] | None = None,
    ) -> EmbeddingsList:
        """Generate embeddings for the provided inputs.

        Args:
            model (EmbedModel | str): Embedding model name.
            inputs (str | list[str] | list[dict[str, Any]]): Text inputs.
                A single string is automatically wrapped.
            parameters (dict[str, Any] | None): Model-specific parameters
                (e.g., ``{"input_type": "passage", "truncate": "END"}``).

        Returns:
            An :class:`EmbeddingsList` with ``.data``, ``.model``, and ``.usage``.

        Raises:
            :exc:`ValidationError`: If *model* is empty.
            :exc:`ValueError`: If *inputs* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> embeddings = pc.inference.embed(
            ...     model="multilingual-e5-large",
            ...     inputs=["Hello, world!"],
            ...     parameters={"input_type": "passage"},
            ... )
            >>> len(embeddings.data)
            1
        """
        require_non_empty("model", str(model))
        normalized_inputs = normalize_embed_inputs(inputs)

        body: dict[str, Any] = {
            "model": str(model),
            "inputs": normalized_inputs,
        }
        if parameters is not None:
            body["parameters"] = parameters

        logger.info("Generating embeddings with model %r", str(model))
        response = self._http.post("/embed", json=body)
        result = self._adapter.to_embeddings_list(response.content)
        logger.debug("Generated %d embeddings", len(result.data))
        return result

    def rerank(
        self,
        model: _enums.RerankModel | str,
        query: str,
        documents: list[str] | list[dict[str, Any]],
        rank_fields: list[str] | None = None,
        return_documents: bool = True,
        top_n: int | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> RerankResult:
        """Rerank documents by relevance to a query.

        Args:
            model (RerankModel | str): Reranking model name.
            query (str): Query text to rank against.
            documents (list[str] | list[dict[str, Any]]): Documents to rank.
                Strings are auto-wrapped as ``{"text": ...}``.
            rank_fields (list[str] | None): Document fields to rank on.
                Defaults to ``["text"]``.
            return_documents (bool): Include document text in response.
                Defaults to ``True``.
            top_n (int | None): Number of top documents to return.
                ``None`` returns all.
            parameters (dict[str, Any] | None): Model-specific parameters.

        Returns:
            A :class:`RerankResult` with ``.data`` and ``.usage``.

        Raises:
            :exc:`ValidationError`: If *model* or *query* is empty.
            :exc:`ValueError`: If *documents* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> result = pc.inference.rerank(
            ...     model="bge-reranker-v2-m3",
            ...     query="Tell me about tech companies",
            ...     documents=["Apple is a fruit.", "Acme Inc. revolutionized tech."],
            ...     top_n=1,
            ... )
            >>> result.data[0].score
            0.95
        """
        require_non_empty("model", str(model))
        require_non_empty("query", query)
        normalized_docs = normalize_rerank_documents(documents)
        rank_fields = rank_fields if rank_fields is not None else ["text"]

        body: dict[str, Any] = {
            "model": str(model),
            "query": query,
            "documents": normalized_docs,
            "rank_fields": rank_fields,
            "return_documents": return_documents,
        }
        if top_n is not None:
            body["top_n"] = top_n
        if parameters is not None:
            body["parameters"] = parameters

        logger.info("Reranking %d documents with model %r", len(normalized_docs), str(model))
        response = self._http.post("/rerank", json=body)
        result = self._adapter.to_rerank_result(response.content)
        logger.debug("Reranked documents, got %d results", len(result.data))
        return result
