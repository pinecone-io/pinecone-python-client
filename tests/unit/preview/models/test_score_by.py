"""Unit tests for score-by query models."""

from __future__ import annotations

import msgspec

from pinecone.models.vectors.sparse import SparseValues
from pinecone.preview.models.score_by import (
    DenseVectorQuery,
    QueryStringQuery,
    ScoreByQuery,
    SparseVectorQuery,
    TextQuery,
)

# ---------------------------------------------------------------------------
# TextQuery
# ---------------------------------------------------------------------------


def test_text_query_wire_shape() -> None:
    q = TextQuery(field="body", query="hello world")
    data = msgspec.json.encode(q)
    decoded = msgspec.json.decode(data)
    assert decoded["type"] == "text"
    assert decoded["field"] == "body"
    assert decoded["query"] == "hello world"


def test_text_query_round_trip() -> None:
    raw = b'{"type": "text", "field": "body", "query": "hello world"}'
    result = msgspec.json.decode(raw, type=TextQuery)
    assert isinstance(result, TextQuery)
    assert result.field == "body"
    assert result.query == "hello world"


def test_text_query_no_extra_fields() -> None:
    q = TextQuery(field="f", query="q")
    assert hasattr(q, "field")
    assert hasattr(q, "query")
    assert not hasattr(q, "values")
    assert not hasattr(q, "sparse_values")


# ---------------------------------------------------------------------------
# QueryStringQuery
# ---------------------------------------------------------------------------


def test_query_string_query_wire_shape() -> None:
    q = QueryStringQuery(query="robots AND adventure")
    data = msgspec.json.encode(q)
    decoded = msgspec.json.decode(data)
    assert decoded["type"] == "query_string"
    assert decoded["query"] == "robots AND adventure"


def test_query_string_query_round_trip() -> None:
    raw = b'{"type": "query_string", "query": "robots AND adventure"}'
    result = msgspec.json.decode(raw, type=QueryStringQuery)
    assert isinstance(result, QueryStringQuery)
    assert result.query == "robots AND adventure"


def test_query_string_query_has_no_field_attr() -> None:
    q = QueryStringQuery(query="test")
    assert not hasattr(q, "field")


# ---------------------------------------------------------------------------
# DenseVectorQuery
# ---------------------------------------------------------------------------


def test_dense_vector_query_wire_shape() -> None:
    q = DenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3])
    data = msgspec.json.encode(q)
    decoded = msgspec.json.decode(data)
    assert decoded["type"] == "dense_vector"
    assert decoded["field"] == "embedding"
    assert decoded["values"] == [0.1, 0.2, 0.3]


def test_dense_vector_query_round_trip() -> None:
    raw = b'{"type": "dense_vector", "field": "embedding", "values": [0.1, 0.2, 0.3]}'
    result = msgspec.json.decode(raw, type=DenseVectorQuery)
    assert isinstance(result, DenseVectorQuery)
    assert result.field == "embedding"
    assert result.values == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# SparseVectorQuery
# ---------------------------------------------------------------------------


def test_sparse_vector_query_wire_shape() -> None:
    sv = SparseValues(indices=[0, 5, 10], values=[0.5, 0.3, 0.8])
    q = SparseVectorQuery(field="_sparse", sparse_values=sv)
    data = msgspec.json.encode(q)
    decoded = msgspec.json.decode(data)
    assert decoded["type"] == "sparse_vector"
    assert decoded["field"] == "_sparse"
    assert decoded["sparse_values"]["indices"] == [0, 5, 10]
    assert decoded["sparse_values"]["values"] == [0.5, 0.3, 0.8]


def test_sparse_vector_query_round_trip() -> None:
    raw = b'{"type": "sparse_vector", "field": "_sparse", "sparse_values": {"indices": [0, 5], "values": [0.5, 0.3]}}'
    result = msgspec.json.decode(raw, type=SparseVectorQuery)
    assert isinstance(result, SparseVectorQuery)
    assert result.field == "_sparse"
    assert result.sparse_values.indices == [0, 5]
    assert result.sparse_values.values == [0.5, 0.3]


# ---------------------------------------------------------------------------
# ScoreByQuery union decode
# ---------------------------------------------------------------------------


def test_union_decodes_text_query() -> None:
    raw = b'{"type": "text", "field": "body", "query": "hello"}'
    result = msgspec.json.decode(raw, type=ScoreByQuery)
    assert isinstance(result, TextQuery)


def test_union_decodes_query_string_query() -> None:
    raw = b'{"type": "query_string", "query": "robots"}'
    result = msgspec.json.decode(raw, type=ScoreByQuery)
    assert isinstance(result, QueryStringQuery)


def test_union_decodes_dense_vector_query() -> None:
    raw = b'{"type": "dense_vector", "field": "emb", "values": [0.1]}'
    result = msgspec.json.decode(raw, type=ScoreByQuery)
    assert isinstance(result, DenseVectorQuery)


def test_union_decodes_sparse_vector_query() -> None:
    raw = b'{"type": "sparse_vector", "field": "_sparse", "sparse_values": {"indices": [1], "values": [0.9]}}'
    result = msgspec.json.decode(raw, type=ScoreByQuery)
    assert isinstance(result, SparseVectorQuery)
