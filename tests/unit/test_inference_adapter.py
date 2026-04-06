"""Tests for the InferenceAdapter and input normalization helpers."""

from __future__ import annotations

import json

import pytest

from pinecone._internal.adapters.inference_adapter import (
    InferenceAdapter,
    normalize_embed_inputs,
    normalize_rerank_documents,
)
from pinecone.models.inference.embed import DenseEmbedding, SparseEmbedding
from pinecone.models.inference.model_list import ModelInfoList
from pinecone.models.inference.models import ModelInfo
from pinecone.models.inference.rerank import RankedDocument, RerankResult


class TestToEmbeddingsListDense:
    def test_to_embeddings_list_dense(self) -> None:
        payload = json.dumps(
            {
                "model": "multilingual-e5-large",
                "vector_type": "dense",
                "data": [
                    {"values": [0.1, 0.2, 0.3], "vector_type": "dense"},
                    {"values": [0.4, 0.5, 0.6], "vector_type": "dense"},
                ],
                "usage": {"total_tokens": 10},
            }
        ).encode()

        result = InferenceAdapter.to_embeddings_list(payload)

        assert result.model == "multilingual-e5-large"
        assert result.vector_type == "dense"
        assert len(result.data) == 2
        assert isinstance(result.data[0], DenseEmbedding)
        assert result.data[0].values == [0.1, 0.2, 0.3]
        assert result.data[1].values == [0.4, 0.5, 0.6]
        assert result.usage.total_tokens == 10


class TestToEmbeddingsListSparse:
    def test_to_embeddings_list_sparse(self) -> None:
        payload = json.dumps(
            {
                "model": "pinecone-sparse-english-v0",
                "vector_type": "sparse",
                "data": [
                    {
                        "sparse_values": [0.5, 0.8],
                        "sparse_indices": [1, 42],
                        "sparse_tokens": ["hello", "world"],
                        "vector_type": "sparse",
                    }
                ],
                "usage": {"total_tokens": 5},
            }
        ).encode()

        result = InferenceAdapter.to_embeddings_list(payload)

        assert result.model == "pinecone-sparse-english-v0"
        assert result.vector_type == "sparse"
        assert len(result.data) == 1
        assert isinstance(result.data[0], SparseEmbedding)
        assert result.data[0].sparse_values == [0.5, 0.8]
        assert result.data[0].sparse_indices == [1, 42]
        assert result.data[0].sparse_tokens == ["hello", "world"]
        assert result.usage.total_tokens == 5


class TestToRerankResult:
    def test_to_rerank_result(self) -> None:
        payload = json.dumps(
            {
                "model": "bge-reranker-v2-m3",
                "data": [
                    {
                        "index": 1,
                        "score": 0.95,
                        "document": {"text": "Relevant doc"},
                    },
                    {"index": 0, "score": 0.3, "document": {"text": "Less relevant"}},
                ],
                "usage": {"rerank_units": 1},
            }
        ).encode()

        result = InferenceAdapter.to_rerank_result(payload)

        assert isinstance(result, RerankResult)
        assert result.model == "bge-reranker-v2-m3"
        assert len(result.data) == 2
        assert isinstance(result.data[0], RankedDocument)
        assert result.data[0].index == 1
        assert result.data[0].score == 0.95
        assert result.data[0].document == {"text": "Relevant doc"}
        assert result.usage.rerank_units == 1


class TestToModelInfo:
    def test_to_model_info(self) -> None:
        payload = json.dumps(
            {
                "model": "multilingual-e5-large",
                "short_description": "A multilingual embedding model.",
                "type": "embed",
                "supported_parameters": [
                    {
                        "parameter": "input_type",
                        "type": "one_of",
                        "value_type": "string",
                        "required": False,
                        "allowed_values": ["passage", "query"],
                    }
                ],
                "vector_type": "dense",
                "default_dimension": 1024,
                "max_sequence_length": 512,
            }
        ).encode()

        result = InferenceAdapter.to_model_info(payload)

        assert isinstance(result, ModelInfo)
        assert result.model == "multilingual-e5-large"
        assert result.short_description == "A multilingual embedding model."
        assert result.type == "embed"
        assert result.vector_type == "dense"
        assert result.default_dimension == 1024
        assert result.max_sequence_length == 512
        assert len(result.supported_parameters) == 1
        assert result.supported_parameters[0].parameter == "input_type"
        assert result.supported_parameters[0].allowed_values == ["passage", "query"]


class TestToModelInfoList:
    def test_to_model_info_list(self) -> None:
        payload = json.dumps(
            {
                "models": [
                    {
                        "model": "multilingual-e5-large",
                        "short_description": "Embedding model",
                        "type": "embed",
                        "supported_parameters": [],
                    },
                    {
                        "model": "bge-reranker-v2-m3",
                        "short_description": "Reranking model",
                        "type": "rerank",
                        "supported_parameters": [],
                    },
                ]
            }
        ).encode()

        result = InferenceAdapter.to_model_info_list(payload)

        assert isinstance(result, ModelInfoList)
        assert len(result) == 2
        assert result.names() == ["multilingual-e5-large", "bge-reranker-v2-m3"]


class TestNormalizeEmbedInputs:
    def test_normalize_embed_inputs_string(self) -> None:
        result = normalize_embed_inputs("hello world")
        assert result == [{"text": "hello world"}]

    def test_normalize_embed_inputs_string_list(self) -> None:
        result = normalize_embed_inputs(["a", "b"])
        assert result == [{"text": "a"}, {"text": "b"}]

    def test_normalize_embed_inputs_dict_list(self) -> None:
        inputs = [{"text": "hello"}, {"text": "world"}]
        result = normalize_embed_inputs(inputs)
        assert result == inputs

    def test_normalize_embed_inputs_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="inputs must not be empty"):
            normalize_embed_inputs([])

    def test_normalize_embed_inputs_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected str, list"):
            normalize_embed_inputs(42)  # type: ignore[arg-type]

    def test_normalize_embed_inputs_list_of_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected list of str or list of dict"):
            normalize_embed_inputs([42, 43])  # type: ignore[list-item]


class TestNormalizeRerankDocuments:
    def test_normalize_rerank_documents_strings(self) -> None:
        result = normalize_rerank_documents(["doc one", "doc two"])
        assert result == [{"text": "doc one"}, {"text": "doc two"}]

    def test_normalize_rerank_documents_dicts(self) -> None:
        docs = [{"text": "doc one"}, {"text": "doc two"}]
        result = normalize_rerank_documents(docs)
        assert result == docs

    def test_normalize_rerank_documents_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="documents must not be empty"):
            normalize_rerank_documents([])
