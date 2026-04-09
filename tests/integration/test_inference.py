"""Integration tests for inference embed and rerank — sync (REST) transport."""

from __future__ import annotations

import pytest

from pinecone import Pinecone

from tests.integration.conftest import (  # noqa: F401 — re-exported for type use
    cleanup_resource,
    poll_until,
    unique_name,
)


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_embed_single_string(client: Pinecone) -> None:
    """embed() with a single string input returns a 1-item EmbeddingsList."""
    result = client.inference.embed(
        model="multilingual-e5-large",
        inputs=["Hello, world!"],
        parameters={"input_type": "passage", "truncate": "END"},
    )
    assert result.model == "multilingual-e5-large"
    assert result.vector_type == "dense"
    assert len(result) == 1
    assert result.usage.total_tokens > 0

    embedding = result.data[0]
    assert isinstance(embedding.values, list)
    assert len(embedding.values) > 0
    assert all(isinstance(v, float) for v in embedding.values)
    assert embedding.vector_type == "dense"


@pytest.mark.integration
def test_embed_multiple_inputs(client: Pinecone) -> None:
    """embed() with multiple inputs returns one embedding per input."""
    inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "Pinecone is a vector database.",
        "Machine learning powers modern AI.",
    ]
    result = client.inference.embed(
        model="multilingual-e5-large",
        inputs=inputs,
        parameters={"input_type": "passage"},
    )
    assert result.model == "multilingual-e5-large"
    assert result.vector_type == "dense"
    assert len(result) == len(inputs)
    assert result.usage.total_tokens > 0

    # All embeddings have the same dimension
    dims = [len(emb.values) for emb in result.data]
    assert len(set(dims)) == 1, f"Expected uniform dimension, got: {dims}"

    # Embeddings are not all zeros
    for emb in result.data:
        assert any(v != 0.0 for v in emb.values)


@pytest.mark.integration
def test_embed_query_vs_passage(client: Pinecone) -> None:
    """embed() with input_type=query vs passage produces different embeddings."""
    text = ["vector databases enable semantic search"]

    query_result = client.inference.embed(
        model="multilingual-e5-large",
        inputs=text,
        parameters={"input_type": "query"},
    )
    passage_result = client.inference.embed(
        model="multilingual-e5-large",
        inputs=text,
        parameters={"input_type": "passage"},
    )

    query_vec = query_result.data[0].values
    passage_vec = passage_result.data[0].values

    # Same length
    assert len(query_vec) == len(passage_vec)

    # Different values (input_type affects the embedding)
    assert query_vec != passage_vec


@pytest.mark.integration
def test_embed_iterable_and_indexable(client: Pinecone) -> None:
    """EmbeddingsList supports len(), iteration, and integer indexing."""
    result = client.inference.embed(
        model="multilingual-e5-large",
        inputs=["foo", "bar"],
        parameters={"input_type": "passage"},
    )
    assert len(result) == 2

    # Integer indexing
    first = result[0]
    second = result[1]
    assert first.values != second.values

    # Iteration
    items = list(result)
    assert len(items) == 2

    # String key access on the list
    assert result["model"] == "multilingual-e5-large"
    assert "model" in result
