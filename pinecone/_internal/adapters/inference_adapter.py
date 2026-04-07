"""Adapter for Inference API responses and request normalization."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone._internal.adapters._decode import convert_response, decode_response
from pinecone.models.inference.embed import (
    DenseEmbedding,
    EmbeddingsList,
    EmbedUsage,
    SparseEmbedding,
)
from pinecone.models.inference.model_list import ModelInfoList
from pinecone.models.inference.models import ModelInfo
from pinecone.models.inference.rerank import RerankResult


class _EmbedEnvelope(Struct, kw_only=True):
    """Internal envelope for the embed response.

    Decodes the top-level fields while keeping data items as raw dicts
    so we can dispatch on vector_type before decoding individual items.
    """

    model: str
    vector_type: str
    data: list[Any]
    usage: EmbedUsage


class _ModelListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-models response."""

    models: list[ModelInfo] = []


class InferenceAdapter:
    """Transforms raw API JSON into inference model instances."""

    @staticmethod
    def to_embeddings_list(data: bytes) -> EmbeddingsList:
        """Decode raw JSON bytes from the embed endpoint into an EmbeddingsList.

        The embed response contains a ``vector_type`` field that determines
        whether items in ``data`` are dense or sparse embeddings. This method
        first decodes the envelope, then converts items to the correct type.
        """
        envelope = decode_response(data, _EmbedEnvelope)

        if envelope.vector_type == "sparse":
            embeddings: list[DenseEmbedding] | list[SparseEmbedding] = [
                convert_response(item, SparseEmbedding) for item in envelope.data
            ]
        else:
            embeddings = [
                convert_response(item, DenseEmbedding) for item in envelope.data
            ]

        return EmbeddingsList(
            model=envelope.model,
            vector_type=envelope.vector_type,
            data=embeddings,
            usage=envelope.usage,
        )

    @staticmethod
    def to_rerank_result(data: bytes) -> RerankResult:
        """Decode raw JSON bytes into a RerankResult."""
        return decode_response(data, RerankResult)

    @staticmethod
    def to_model_info(data: bytes) -> ModelInfo:
        """Decode raw JSON bytes into a ModelInfo."""
        return decode_response(data, ModelInfo)

    @staticmethod
    def to_model_info_list(data: bytes) -> ModelInfoList:
        """Decode raw JSON bytes from the list-models endpoint into a ModelInfoList."""
        envelope = decode_response(data, _ModelListEnvelope)
        return ModelInfoList(envelope.models)


def normalize_embed_inputs(
    inputs: str | list[str] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize embed inputs into the API's expected format.

    Args:
        inputs: A single string, list of strings, or list of dicts.

    Returns:
        A list of dicts with ``"text"`` keys.

    Raises:
        ValueError: If inputs is an empty list.
        TypeError: If inputs is not a recognized type or contains mixed types.
    """
    if isinstance(inputs, str):
        return [{"text": inputs}]
    if isinstance(inputs, list):
        if len(inputs) == 0:
            raise ValueError("inputs must not be empty")
        if not all(isinstance(item, (str, dict)) for item in inputs):
            raise TypeError("each input must be a string or dictionary")
        first = inputs[0]
        if isinstance(first, str):
            if not all(isinstance(item, str) for item in inputs):
                raise TypeError("each input must be a string or dictionary")
            str_inputs: list[str] = inputs  # type: ignore[assignment]
            return [{"text": s} for s in str_inputs]
        if isinstance(first, dict):
            if not all(isinstance(item, dict) for item in inputs):
                raise TypeError("each input must be a string or dictionary")
            return inputs  # type: ignore[return-value]
        raise TypeError(
            f"Expected list of str or list of dict, got list of {type(first).__name__}"
        )
    raise TypeError(f"Expected str, list[str], or list[dict], got {type(inputs).__name__}")


def normalize_rerank_documents(
    documents: list[str] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize rerank documents into the API's expected format.

    Args:
        documents: A list of strings or list of dicts.

    Returns:
        A list of dicts with ``"text"`` keys.

    Raises:
        TypeError: If documents is not a list or contains invalid element types.
        ValueError: If documents is empty.
    """
    if not isinstance(documents, list):
        raise TypeError("documents must be a list of strings or list of dictionaries")
    if len(documents) == 0:
        raise ValueError("documents must not be empty")
    if not all(isinstance(d, (str, dict)) for d in documents):
        raise TypeError("each document must be a string or dictionary")
    if isinstance(documents[0], str):
        if not all(isinstance(d, str) for d in documents):
            raise TypeError("each document must be a string or dictionary")
        return [{"text": s} for s in documents]
    if isinstance(documents[0], dict):
        if not all(isinstance(d, dict) for d in documents):
            raise TypeError("each document must be a string or dictionary")
    return documents  # type: ignore[return-value]
