"""Integration tests for inference embed and rerank — sync (REST) transport."""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.errors import PineconeTypeError, PineconeValueError
from pinecone.models.inference.embed import SparseEmbedding
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
def test_embed_sparse_model_returns_sparse_embeddings(client: Pinecone) -> None:
    """embed() with pinecone-sparse-english-v0 returns SparseEmbedding objects.

    Verifies unified-enum-0006 (sparse model is known/available) and exercises
    the sparse branch in InferenceAdapter.to_embeddings_list():

        if envelope.vector_type == "sparse":
            embeddings = [SparseEmbedding(...) for item in envelope.data]

    All existing embed tests use multilingual-e5-large (dense), so this branch
    has never been exercised in integration tests before.
    """
    inputs = [
        "What is vector search?",
        "Pinecone is a managed vector database.",
    ]
    result = client.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=inputs,
        parameters={"input_type": "passage"},
    )

    # Top-level response shape
    assert result.model == "pinecone-sparse-english-v0"
    assert result.vector_type == "sparse"
    assert len(result) == len(inputs)
    assert result.usage.total_tokens > 0

    # Each item is a SparseEmbedding with non-empty parallel arrays
    for emb in result:
        assert isinstance(emb, SparseEmbedding)
        assert emb.vector_type == "sparse"
        assert isinstance(emb.sparse_values, list)
        assert isinstance(emb.sparse_indices, list)
        assert len(emb.sparse_values) > 0
        assert len(emb.sparse_indices) > 0
        # Parallel arrays must have the same length
        assert len(emb.sparse_values) == len(emb.sparse_indices)
        assert all(isinstance(v, float) for v in emb.sparse_values)
        assert all(isinstance(i, int) for i in emb.sparse_indices)
        # Without return_tokens, sparse_tokens is None
        assert emb.sparse_tokens is None


@pytest.mark.integration
def test_embed_list_of_dict_inputs(client: Pinecone) -> None:
    """embed() accepts a list of dicts as inputs and returns one embedding per dict.

    Verifies unified-inf-0009: "The embed operation accepts a list of strings,
    a list of dictionaries, or a single bare string as inputs."

    All existing embed integration tests pass strings. This test exercises the
    dict pass-through path in normalize_embed_inputs(), where list[dict] is sent
    to the API as-is (each dict must have a "text" key). The unit tests cover
    the normalization logic; this confirms the dict format is accepted end-to-end
    by the real API.
    """
    inputs = [
        {"text": "What is a vector database?"},
        {"text": "Pinecone is a managed vector database service."},
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

    for emb in result:
        assert isinstance(emb.values, list)
        assert len(emb.values) > 0
        assert all(isinstance(v, float) for v in emb.values)

    # Two different texts produce different embeddings
    assert result.data[0].values != result.data[1].values


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


@pytest.mark.integration
def test_embed_bare_string_auto_wrapped_rest(client: Pinecone) -> None:
    """embed() with a single bare string (not a list) is auto-wrapped and returns 1 embedding.

    Verifies unified-inf-0008: The embed operation accepts a single string input
    and wraps it into a list automatically.  The normalize_embed_inputs() adapter
    has a dedicated branch: `if isinstance(inputs, str): return [{"text": inputs}]`.
    All other integration tests pass a list (even single-element lists), so that
    branch has never been exercised end-to-end against the live API.
    """
    # Pass a bare string — NOT wrapped in a list
    result = client.inference.embed(
        model="multilingual-e5-large",
        inputs="What is a vector database?",
        parameters={"input_type": "passage", "truncate": "END"},
    )

    # The bare string should have been auto-wrapped → 1 embedding returned
    assert result.model == "multilingual-e5-large"
    assert result.vector_type == "dense"
    assert len(result) == 1, "Bare string should be auto-wrapped to a single-element batch"
    assert result.usage.total_tokens > 0

    embedding = result.data[0]
    assert isinstance(embedding.values, list)
    assert len(embedding.values) > 0
    assert all(isinstance(v, float) for v in embedding.values)
    assert embedding.vector_type == "dense"

    # The result should be identical to passing the same string in a list
    result_as_list = client.inference.embed(
        model="multilingual-e5-large",
        inputs=["What is a vector database?"],
        parameters={"input_type": "passage", "truncate": "END"},
    )
    assert len(result_as_list) == 1
    # Same string → same embedding values (deterministic model)
    assert result.data[0].values == result_as_list.data[0].values


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_rerank_basic(client: Pinecone) -> None:
    """rerank() returns a RerankResult with ranked documents sorted by score."""
    query = "What is machine learning?"
    documents = [
        "Paris is the capital of France.",
        "Machine learning is a subset of AI that uses algorithms to learn from data.",
        "Deep learning uses neural networks with many layers.",
    ]
    result = client.inference.rerank(
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
def test_rerank_with_top_n(client: Pinecone) -> None:
    """rerank() with top_n limits the number of returned results."""
    documents = [
        "Machine learning is a branch of artificial intelligence.",
        "Paris is known for the Eiffel Tower.",
        "Neural networks mimic the human brain.",
        "The ocean covers 71% of Earth's surface.",
    ]
    result = client.inference.rerank(
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
def test_rerank_return_documents_false(client: Pinecone) -> None:
    """rerank() with return_documents=False omits document text from results."""
    documents = [
        "The sky is blue due to Rayleigh scattering.",
        "Vector databases store high-dimensional embeddings.",
    ]
    result = client.inference.rerank(
        model="bge-reranker-v2-m3",
        query="How do vector databases work?",
        documents=documents,
        return_documents=False,
    )

    assert len(result.data) == len(documents)
    for ranked in result.data:
        assert ranked.document is None


@pytest.mark.integration
def test_rerank_string_inputs_auto_wrapped(client: Pinecone) -> None:
    """rerank() with plain string documents auto-wraps them as {text: ...}."""
    documents = [
        "Machine learning enables computers to learn from experience.",
        "Cooking pasta requires boiling water.",
    ]
    result = client.inference.rerank(
        model="bge-reranker-v2-m3",
        query="How does machine learning work?",
        documents=documents,
        return_documents=True,
    )

    assert len(result.data) == len(documents)
    for ranked in result.data:
        # SDK wraps strings as {"text": ...} before sending
        assert ranked.document is not None
        assert "text" in ranked.document


# ---------------------------------------------------------------------------
# list_models / get_model
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_models_returns_nonempty_list(client: Pinecone) -> None:
    """list_models() returns a ModelInfoList with at least one known model."""
    result = client.inference.list_models()

    assert len(result) > 0
    names = result.names()
    assert isinstance(names, list)
    assert "multilingual-e5-large" in names


@pytest.mark.integration
def test_list_models_supports_iteration_and_indexing(client: Pinecone) -> None:
    """ModelInfoList supports len(), iteration, integer indexing, and 'models' key."""
    result = client.inference.list_models()

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
def test_list_models_filter_by_type_embed(client: Pinecone) -> None:
    """list_models(type='embed') returns only embed models."""
    result = client.inference.list_models(type="embed")

    assert len(result) > 0
    for model_info in result:
        assert model_info.type == "embed"


@pytest.mark.integration
def test_list_models_filter_by_type_rerank(client: Pinecone) -> None:
    """list_models(type='rerank') returns only rerank models."""
    result = client.inference.list_models(type="rerank")

    assert len(result) > 0
    for model_info in result:
        assert model_info.type == "rerank"


@pytest.mark.integration
def test_get_model_returns_model_info(client: Pinecone) -> None:
    """get_model() returns a ModelInfo with name, vector_type, and default_dimension."""
    model_info = client.inference.get_model(model="multilingual-e5-large")

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
def test_get_model_rerank_model(client: Pinecone) -> None:
    """get_model() works for rerank models; vector_type and default_dimension are None."""
    model_info = client.inference.get_model(model="bge-reranker-v2-m3")

    assert model_info.model == "bge-reranker-v2-m3"
    assert model_info.type == "rerank"
    # Rerank models don't produce vectors
    assert model_info.vector_type is None
    assert model_info.default_dimension is None


# ---------------------------------------------------------------------------
# rerank — input validation (client-side, no API call)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_rerank_documents_validation_rest(client: Pinecone) -> None:
    """rerank() raises PineconeValueError for empty docs and PineconeTypeError for non-list.

    Verifies unified-inf-0018 (empty list rejected) and unified-inf-0019 (non-list rejected).
    These validations happen client-side before any HTTP request is made.
    """
    # unified-inf-0018: empty list must be rejected
    with pytest.raises(PineconeValueError):
        client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=[],
        )

    # unified-inf-0019: non-list documents (plain string) must be rejected
    with pytest.raises(PineconeTypeError):
        client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents="not a list",  # type: ignore[arg-type]
        )

    # unified-inf-0019: non-list documents (integer) must be rejected
    with pytest.raises(PineconeTypeError):
        client.inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=42,  # type: ignore[arg-type]
        )


@pytest.mark.integration
def test_rerank_dict_documents_with_rank_fields(client: Pinecone) -> None:
    """rerank() with list-of-dict documents and custom rank_fields passes dicts through.

    Verifies unified-inf-0010 (dict document path): when documents are passed as dicts,
    they are sent to the API without wrapping, and rank_fields selects which field to rank on.
    All existing integration tests use string inputs (auto-wrapped). This test exercises
    the dict code path in normalize_rerank_documents().
    """
    query = "machine learning and artificial intelligence"
    documents = [
        {"body": "The Eiffel Tower is a famous landmark in Paris, France.", "category": "travel"},
        {"body": "Machine learning is a branch of AI that learns from data.", "category": "tech"},
        {"body": "Neural networks are inspired by the human brain.", "category": "tech"},
        {"body": "The best pizza is from Naples, Italy.", "category": "food"},
    ]

    result = client.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=documents,
        rank_fields=["body"],  # non-default field — default is ["text"]
        return_documents=True,
    )

    assert result.model == "bge-reranker-v2-m3"
    assert len(result.data) == len(documents)
    assert result.usage.rerank_units > 0

    # Scores must be sorted descending
    scores = [item.score for item in result.data]
    assert scores == sorted(scores, reverse=True), f"Scores not sorted descending: {scores}"

    # Each ranked result must have an index into the original list and a document
    for ranked in result.data:
        assert isinstance(ranked.index, int)
        assert 0 <= ranked.index < len(documents)
        assert isinstance(ranked.score, float)
        assert ranked.document is not None

        # Dict documents are passed through as-is (not wrapped in {"text": ...})
        # so the original "body" and "category" keys must be present in the returned doc
        assert "body" in ranked.document, (
            f"'body' key missing from ranked.document: {ranked.document}"
        )
        assert "category" in ranked.document, (
            f"'category' key missing from ranked.document: {ranked.document}"
        )

    # The most relevant document should be one of the AI/ML ones (indexes 1 or 2)
    assert result.data[0].index in {1, 2}, (
        f"Expected AI/ML doc to rank first, got index {result.data[0].index}"
    )


# ---------------------------------------------------------------------------
# embed — input validation (client-side, no API call)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_embed_inputs_validation_rest(client: Pinecone) -> None:
    """embed() raises PineconeValueError for empty inputs and PineconeTypeError for wrong type.

    Verifies unified-inf-0016 (empty list rejected) and unified-inf-0017 (non-list rejected).
    These validations fire client-side in normalize_embed_inputs() before any HTTP request.
    """
    model = "multilingual-e5-large"

    # unified-inf-0016: empty list must be rejected
    with pytest.raises(PineconeValueError):
        client.inference.embed(model=model, inputs=[])

    # unified-inf-0017: plain integer rejected
    with pytest.raises(PineconeTypeError):
        client.inference.embed(model=model, inputs=42)  # type: ignore[arg-type]

    # unified-inf-0017: tuple rejected (not str or list)
    with pytest.raises(PineconeTypeError):
        client.inference.embed(model=model, inputs=("a", "b"))  # type: ignore[arg-type]

    # unified-inf-0017: mixed list (string + integer) rejected
    with pytest.raises(PineconeTypeError):
        client.inference.embed(model=model, inputs=["valid", 999])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# list_models — vector_type filter and invalid-value validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_models_filter_by_vector_type_and_invalid_values(client: Pinecone) -> None:
    """list_models() supports vector_type filter and rejects invalid type/vector_type values.

    Verifies unified-inf-0020: the list-models operation accepts only "embed" and "rerank"
    as the type filter, and only "dense" and "sparse" as the vector-type filter. Invalid
    values are rejected client-side with PineconeValueError before any HTTP request.
    """
    # vector_type="dense" → only dense embed models returned
    dense_result = client.inference.list_models(vector_type="dense")
    assert len(dense_result) > 0
    for model_info in dense_result:
        assert model_info.vector_type == "dense", (
            f"Expected vector_type='dense', got {model_info.vector_type!r} for model {model_info.model!r}"
        )

    # vector_type="sparse" → only sparse embed models returned
    sparse_result = client.inference.list_models(vector_type="sparse")
    assert len(sparse_result) > 0
    for model_info in sparse_result:
        assert model_info.vector_type == "sparse", (
            f"Expected vector_type='sparse', got {model_info.vector_type!r} for model {model_info.model!r}"
        )

    # invalid type → PineconeValueError, no HTTP call
    with pytest.raises(PineconeValueError):
        client.inference.list_models(type="invalid_type")

    # invalid vector_type → PineconeValueError, no HTTP call
    with pytest.raises(PineconeValueError):
        client.inference.list_models(vector_type="invalid_vector_type")


# ---------------------------------------------------------------------------
# model dict-like access error behaviors on real API responses
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_model_dict_keyerror_and_readonly_rest(client: Pinecone) -> None:
    """Dict-like access on model objects from real API responses raises correct errors.

    Uses list_models() (no resources created or cleaned up) to obtain a real
    ModelInfo object deserialized from the live API, then verifies:

    - unified-model-0007: model["nonexistent_key"] raises KeyError
    - unified-model-0010: model["model"] = "x" raises TypeError (read-only)

    Also verifies positive behaviors work on real deserialized data:
    - model["model"] returns a string (existing key subscript access)
    - "model" in model → True
    - "totally_absent_key" in model → False
    """
    models = client.inference.list_models()
    assert len(models) > 0, "list_models() must return at least one model for this test"

    model_info = models[0]

    # --- unified-model-0007: KeyError on non-existent key ---
    with pytest.raises(KeyError):
        _ = model_info["nonexistent_key_xyz"]

    # --- unified-model-0010: no item assignment (read-only) ---
    with pytest.raises(TypeError):
        model_info["model"] = "injected"  # type: ignore[index]

    # Positive: existing-key access and membership tests work on real data
    assert isinstance(model_info["model"], str)
    assert len(model_info["model"]) > 0
    assert "model" in model_info
    assert "totally_absent_key_xyz" not in model_info
