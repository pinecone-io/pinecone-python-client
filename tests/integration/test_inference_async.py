"""Integration tests for inference embed and rerank — async (REST) transport."""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone
from pinecone.errors import PineconeTypeError, PineconeValueError
from pinecone.models.inference.embed import SparseEmbedding

# ---------------------------------------------------------------------------
# embed (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_single_string_async(async_client: AsyncPinecone) -> None:
    """async embed() with a single string input returns a 1-item EmbeddingsList."""
    result = await async_client.inference.embed(
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
@pytest.mark.asyncio
async def test_embed_sparse_model_returns_sparse_embeddings_async(
    async_client: AsyncPinecone,
) -> None:
    """async embed() with pinecone-sparse-english-v0 returns SparseEmbedding objects.

    Async variant of test_embed_sparse_model_returns_sparse_embeddings. Verifies
    unified-enum-0006 and the sparse decode path in InferenceAdapter.
    """
    inputs = [
        "What is vector search?",
        "Pinecone is a managed vector database.",
    ]
    result = await async_client.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=inputs,
        parameters={"input_type": "passage"},
    )

    assert result.model == "pinecone-sparse-english-v0"
    assert result.vector_type == "sparse"
    assert len(result) == len(inputs)
    assert result.usage.total_tokens > 0

    for emb in result:
        assert isinstance(emb, SparseEmbedding)
        assert emb.vector_type == "sparse"
        assert isinstance(emb.sparse_values, list)
        assert isinstance(emb.sparse_indices, list)
        assert len(emb.sparse_values) > 0
        assert len(emb.sparse_indices) > 0
        assert len(emb.sparse_values) == len(emb.sparse_indices)
        assert all(isinstance(v, float) for v in emb.sparse_values)
        assert all(isinstance(i, int) for i in emb.sparse_indices)
        assert emb.sparse_tokens is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_list_of_dict_inputs_async(async_client: AsyncPinecone) -> None:
    """async embed() accepts a list of dicts as inputs and returns one embedding per dict.

    Async variant of test_embed_list_of_dict_inputs. Verifies unified-inf-0009:
    the embed operation accepts list[dict] inputs end-to-end with the real API.
    """
    inputs = [
        {"text": "What is a vector database?"},
        {"text": "Pinecone is a managed vector database service."},
    ]
    result = await async_client.inference.embed(
        model="multilingual-e5-large",
        inputs=inputs,
        parameters={"input_type": "passage"},
    )

    assert result.model == "multilingual-e5-large"
    assert result.vector_type == "dense"
    assert len(result) == len(inputs)
    assert result.usage.total_tokens > 0

    for emb in result:
        assert isinstance(emb.values, list)
        assert len(emb.values) > 0
        assert all(isinstance(v, float) for v in emb.values)

    assert result.data[0].values != result.data[1].values


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_multiple_inputs_async(async_client: AsyncPinecone) -> None:
    """async embed() with multiple inputs returns one embedding per input."""
    inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "Pinecone is a vector database.",
        "Machine learning powers modern AI.",
    ]
    result = await async_client.inference.embed(
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
@pytest.mark.asyncio
async def test_embed_iterable_and_indexable_async(async_client: AsyncPinecone) -> None:
    """async: EmbeddingsList supports len(), iteration, and integer indexing."""
    result = await async_client.inference.embed(
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


# ---------------------------------------------------------------------------
# rerank (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rerank_basic_async(async_client: AsyncPinecone) -> None:
    """async rerank() returns a RerankResult with ranked documents sorted by score."""
    query = "What is machine learning?"
    documents = [
        "Paris is the capital of France.",
        "Machine learning is a subset of AI that uses algorithms to learn from data.",
        "Deep learning uses neural networks with many layers.",
    ]
    result = await async_client.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=documents,
    )

    assert result.model == "bge-reranker-v2-m3"
    assert len(result.data) == len(documents)
    assert result.usage.rerank_units > 0

    # Results are sorted by score descending
    scores = [doc.score for doc in result.data]
    assert scores == sorted(scores, reverse=True), f"Scores not sorted descending: {scores}"

    # Each result has index, score, and document
    for ranked in result.data:
        assert isinstance(ranked.index, int)
        assert 0 <= ranked.index < len(documents)
        assert isinstance(ranked.score, float)
        assert ranked.document is not None
        assert "text" in ranked.document

    # The most relevant document should be the ML one (original index 1)
    assert result.data[0].index == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rerank_with_top_n_async(async_client: AsyncPinecone) -> None:
    """async rerank() with top_n limits the number of returned results."""
    documents = [
        "Machine learning is a branch of artificial intelligence.",
        "Paris is known for the Eiffel Tower.",
        "Neural networks mimic the human brain.",
        "The ocean covers 71% of Earth's surface.",
    ]
    result = await async_client.inference.rerank(
        model="bge-reranker-v2-m3",
        query="Tell me about AI and neural networks",
        documents=documents,
        top_n=2,
    )

    assert result.model == "bge-reranker-v2-m3"
    assert len(result.data) == 2
    assert result.usage.rerank_units > 0

    scores = [doc.score for doc in result.data]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rerank_return_documents_false_async(async_client: AsyncPinecone) -> None:
    """async rerank() with return_documents=False omits document text from results."""
    documents = [
        "The sky is blue due to Rayleigh scattering.",
        "Vector databases store high-dimensional embeddings.",
    ]
    result = await async_client.inference.rerank(
        model="bge-reranker-v2-m3",
        query="How do vector databases work?",
        documents=documents,
        return_documents=False,
    )

    assert len(result.data) == len(documents)
    for ranked in result.data:
        assert ranked.document is None


# ---------------------------------------------------------------------------
# list_models / get_model (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_models_returns_nonempty_list_async(async_client: AsyncPinecone) -> None:
    """async list_models() returns a ModelInfoList with at least one known model."""
    result = await async_client.inference.list_models()

    assert len(result) > 0
    names = result.names()
    assert isinstance(names, list)
    assert "multilingual-e5-large" in names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_models_supports_iteration_and_indexing_async(async_client: AsyncPinecone) -> None:
    """async: ModelInfoList supports len(), iteration, integer indexing, and 'models' key."""
    result = await async_client.inference.list_models()

    # Integer indexing
    first = result[0]
    assert isinstance(first.model, str)
    assert len(first.model) > 0

    # Iteration produces ModelInfo objects
    items = list(result)
    assert len(items) == len(result)
    assert all(hasattr(m, "model") for m in items)

    # String key access
    models_list = result["models"]
    assert isinstance(models_list, list)
    assert len(models_list) == len(result)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_models_filter_by_type_embed_async(async_client: AsyncPinecone) -> None:
    """async list_models(type='embed') returns only embed models."""
    result = await async_client.inference.list_models(type="embed")

    assert len(result) > 0
    for model_info in result:
        assert model_info.type == "embed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_models_filter_by_type_rerank_async(async_client: AsyncPinecone) -> None:
    """async list_models(type='rerank') returns only rerank models."""
    result = await async_client.inference.list_models(type="rerank")

    assert len(result) > 0
    for model_info in result:
        assert model_info.type == "rerank"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_model_returns_model_info_async(async_client: AsyncPinecone) -> None:
    """async get_model() returns a ModelInfo with name, vector_type, and default_dimension."""
    model_info = await async_client.inference.get_model(model="multilingual-e5-large")

    # Required fields
    assert model_info.model == "multilingual-e5-large"
    assert model_info.type == "embed"

    # Alias property
    assert model_info.name == "multilingual-e5-large"

    # Embed-specific fields
    assert model_info.vector_type is not None
    assert model_info.default_dimension is not None
    assert isinstance(model_info.default_dimension, int)
    assert model_info.default_dimension > 0

    # Bracket access
    assert model_info["model"] == "multilingual-e5-large"
    assert model_info["name"] == "multilingual-e5-large"  # alias
    assert "model" in model_info
    assert "name" in model_info  # alias


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_model_rerank_model_async(async_client: AsyncPinecone) -> None:
    """async get_model() works for rerank models; vector_type and default_dimension are None."""
    model_info = await async_client.inference.get_model(model="bge-reranker-v2-m3")

    assert model_info.model == "bge-reranker-v2-m3"
    assert model_info.type == "rerank"
    # Rerank models don't produce vectors
    assert model_info.vector_type is None
    assert model_info.default_dimension is None


# ---------------------------------------------------------------------------
# rerank — input validation (async, client-side, no API call)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rerank_documents_validation_rest_async(async_client: AsyncPinecone) -> None:
    """async rerank() raises PineconeValueError for empty docs, PineconeTypeError for non-list.

    Verifies unified-inf-0018 (empty list rejected) and unified-inf-0019 (non-list rejected).
    These validations happen client-side before any HTTP request is made.
    """
    # unified-inf-0018: empty list must be rejected
    with pytest.raises(PineconeValueError):
        await async_client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=[],
        )

    # unified-inf-0019: non-list documents (plain string) must be rejected
    with pytest.raises(PineconeTypeError):
        await async_client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents="not a list",  # type: ignore[arg-type]
        )

    # unified-inf-0019: non-list documents (integer) must be rejected
    with pytest.raises(PineconeTypeError):
        await async_client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=42,  # type: ignore[arg-type]
        )
