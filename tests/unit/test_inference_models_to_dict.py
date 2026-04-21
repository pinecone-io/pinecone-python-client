"""Unit tests for to_dict() on inference data models."""

from __future__ import annotations

from pinecone.models.inference.embed import DenseEmbedding, EmbedUsage, SparseEmbedding
from pinecone.models.inference.rerank import RankedDocument, RerankUsage


def test_embed_usage_to_dict() -> None:
    result = EmbedUsage(total_tokens=100).to_dict()
    assert result == {"total_tokens": 100}


def test_dense_embedding_to_dict_required_fields() -> None:
    result = DenseEmbedding(values=[0.1, 0.2]).to_dict()
    assert result == {"values": [0.1, 0.2], "vector_type": "dense"}


def test_dense_embedding_to_dict_empty_list() -> None:
    result = DenseEmbedding(values=[]).to_dict()
    assert result == {"values": [], "vector_type": "dense"}


def test_sparse_embedding_to_dict() -> None:
    result = SparseEmbedding(sparse_values=[0.5], sparse_indices=[42]).to_dict()
    assert set(result.keys()) == {"sparse_values", "sparse_indices", "sparse_tokens", "vector_type"}
    assert result["sparse_values"] == [0.5]
    assert result["sparse_indices"] == [42]
    assert result["vector_type"] == "sparse"


def test_sparse_embedding_to_dict_tokens_none() -> None:
    result = SparseEmbedding(sparse_values=[0.5], sparse_indices=[42], sparse_tokens=None).to_dict()
    assert "sparse_tokens" in result
    assert result["sparse_tokens"] is None


def test_rerank_usage_to_dict() -> None:
    result = RerankUsage(rerank_units=3).to_dict()
    assert result == {"rerank_units": 3}


def test_ranked_document_to_dict_required_only() -> None:
    result = RankedDocument(index=0, score=0.9).to_dict()
    assert result == {"index": 0, "score": 0.9, "document": None}


def test_ranked_document_to_dict_with_document() -> None:
    result = RankedDocument(index=1, score=0.8, document={"text": "hello"}).to_dict()
    assert result["document"] == {"text": "hello"}


def test_to_dict_is_pure_read() -> None:
    dense = DenseEmbedding(values=[1.0, 2.0])
    d = dense.to_dict()
    d["values"] = [99.0]
    assert dense.values == [1.0, 2.0]

    doc = RankedDocument(index=0, score=0.5, document={"text": "original"})
    d2 = doc.to_dict()
    d2["index"] = 999
    assert doc.index == 0
