"""Integration tests for search-records with integrated inference — sync (REST + gRPC).

Covers:
  - search-records: basic text search
  - search-with-rerank: text search with inline reranking
  - search-by-id: search using a stored record ID as the query vector
  - search-with-filter: text search with a metadata filter expression
"""

from __future__ import annotations

import pytest

from pinecone import EmbedConfig, IntegratedSpec, Pinecone
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.vectors.search import Hit, RerankConfig, SearchInputs, SearchRecordsResponse, SearchResult, SearchUsage
from tests.integration.conftest import cleanup_resource, poll_until, unique_name, wait_for_ready

# ---------------------------------------------------------------------------
# search-records — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_records_rest(client: Pinecone, api_key: str) -> None:
    """search() with text inputs on an integrated index returns SearchRecordsResponse with hits."""
    name = unique_name("idx")
    namespace = "srch-ns"
    try:
        client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "text"},
                ),
            ),
        )

        # Wait for the index to become ready using SDK's describe
        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        index = client.index(name=name)

        # Upsert records: text fields are embedded server-side
        index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "doc-1", "text": "Vector databases enable fast similarity search."},
                {"_id": "doc-2", "text": "RAG combines retrieval with language model generation."},
                {"_id": "doc-3", "text": "Embeddings are dense vector representations of data."},
            ],
        )

        # Wait for records to be searchable (eventual consistency)
        response = poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=3,
                inputs={"text": "similarity search with embeddings"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="records searchable after upsert",
        )

        assert isinstance(response, SearchRecordsResponse)
        assert isinstance(response.result, SearchResult)
        assert len(response.result.hits) > 0
        assert len(response.result.hits) <= 3

        # Verify hit structure
        for hit in response.result.hits:
            assert isinstance(hit, Hit)
            assert isinstance(hit.id, str)
            assert isinstance(hit.score, float)
            assert hit.id.startswith("doc-")

        # Usage: read_units and embed_total_tokens are set for text-input searches
        assert isinstance(response.usage, SearchUsage)
        assert response.usage.read_units > 0
        assert response.usage.embed_total_tokens is not None
        assert response.usage.embed_total_tokens > 0

        # Convenience property aliases work
        first = response.result.hits[0]
        assert first.id == first.id_
        assert first.score == first.score_

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# search-records — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_records_grpc(client: Pinecone, api_key: str) -> None:
    """search() via GrpcIndex with integrated inference."""
    name = unique_name("idx")
    namespace = "srch-ns"
    try:
        client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "text"},
                ),
            ),
        )
        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )
        # GrpcIndex creation raises ModuleNotFoundError (IT-0002)
        index = client.index(name=name, grpc=True)

        index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "doc-1", "text": "Vector databases enable fast similarity search."},
            ],
        )

        response = index.search(
            namespace=namespace,
            top_k=1,
            inputs={"text": "similarity search"},
        )
        assert len(response.result.hits) > 0
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# search-with-rerank — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_with_rerank_rest(client: Pinecone, api_key: str) -> None:
    """search() with inline rerank parameter re-ranks hits and populates usage.rerank_units."""
    name = unique_name("idx")
    namespace = "rerank-ns"
    try:
        client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "text"},
                ),
            ),
        )

        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        index = client.index(name=name)

        # Upsert records with varied text content
        index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "rr-1", "text": "Vector databases enable fast similarity search at scale."},
                {"_id": "rr-2", "text": "RAG combines retrieval with language model generation."},
                {"_id": "rr-3", "text": "Embeddings are dense vector representations of text data."},
                {"_id": "rr-4", "text": "Python is a popular programming language for AI projects."},
                {"_id": "rr-5", "text": "Pinecone provides serverless vector database infrastructure."},
            ],
        )

        # Wait for records to be searchable (eventual consistency)
        poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=5,
                inputs={"text": "vector database similarity search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="records searchable before rerank test",
        )

        # Search with inline reranking — top_n=3 limits results after reranking
        response = index.search(
            namespace=namespace,
            top_k=5,
            inputs={"text": "vector database similarity search"},
            rerank={
                "model": "bge-reranker-v2-m3",
                "rank_fields": ["text"],
                "top_n": 3,
            },
        )

        assert isinstance(response, SearchRecordsResponse)
        # top_n=3 caps the number of hits after reranking
        assert len(response.result.hits) > 0
        assert len(response.result.hits) <= 3

        # Each hit has id and score
        for hit in response.result.hits:
            assert isinstance(hit, Hit)
            assert isinstance(hit.id, str)
            assert isinstance(hit.score, float)

        # Rerank usage should be populated
        assert isinstance(response.usage, SearchUsage)
        assert response.usage.rerank_units is not None
        assert response.usage.rerank_units > 0

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# search-by-id — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_by_id_rest(client: Pinecone, api_key: str) -> None:
    """search(id=...) uses a stored record's embedding as the query vector.

    Verifies that search-by-id returns a SearchRecordsResponse with the same
    structure as search-by-text: hits list, hit.id (str), hit.score (float),
    and usage.read_units > 0.
    """
    name = unique_name("idx")
    namespace = "sid-ns"
    try:
        client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "text"},
                ),
            ),
        )

        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        index = client.index(name=name)

        # Upsert records — embeddings are generated server-side from the text field
        index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "sid-1", "text": "Vector databases enable fast similarity search."},
                {"_id": "sid-2", "text": "RAG combines retrieval with language model generation."},
                {"_id": "sid-3", "text": "Embeddings are dense vector representations of data."},
            ],
        )

        # Wait until at least one record is searchable (eventual consistency)
        poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=3,
                inputs={"text": "similarity search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="records searchable before search-by-id test",
        )

        # Search using an existing record ID as the query seed
        response = index.search(namespace=namespace, top_k=3, id="sid-1")

        assert isinstance(response, SearchRecordsResponse)
        assert isinstance(response.result, SearchResult)
        assert len(response.result.hits) > 0
        assert len(response.result.hits) <= 3

        # Verify hit structure is identical to search-by-text
        for hit in response.result.hits:
            assert isinstance(hit, Hit)
            assert isinstance(hit.id, str)
            assert isinstance(hit.score, float)

        # Usage should reflect a read operation
        assert isinstance(response.usage, SearchUsage)
        assert response.usage.read_units > 0

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# search-with-filter — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_with_filter_rest(client: Pinecone, api_key: str) -> None:
    """search() with a metadata filter returns only records matching the filter.

    Upserts records with two distinct category values ('science' and 'history').
    A filter for category=science must return only science records; no history
    record should appear. The fields parameter is used to verify category values
    in the returned hits.
    """
    name = unique_name("idx")
    namespace = "swf-ns"
    try:
        client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "text"},
                ),
            ),
        )

        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        index = client.index(name=name)

        # Upsert records with a 'category' metadata field for filtering
        index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "swf-sci-1", "text": "Quantum mechanics describes subatomic particles.", "category": "science"},
                {"_id": "swf-sci-2", "text": "DNA encodes the genetic information of organisms.", "category": "science"},
                {"_id": "swf-sci-3", "text": "Gravity is a fundamental force in physics.", "category": "science"},
                {"_id": "swf-hist-1", "text": "The Roman Empire lasted for centuries.", "category": "history"},
                {"_id": "swf-hist-2", "text": "The Renaissance was a cultural movement in Europe.", "category": "history"},
            ],
        )

        # Wait for records to be searchable (eventual consistency)
        poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=5,
                inputs={"text": "science and research"},
            ),
            check_fn=lambda r: len(r.result.hits) >= 3,
            timeout=120,
            description="all records searchable before filter test",
        )

        # Search with category=science filter, requesting category field in hits
        response = index.search(
            namespace=namespace,
            top_k=5,
            inputs={"text": "science and research"},
            filter={"category": {"$eq": "science"}},
            fields=["category"],
        )

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) > 0
        # Must not exceed the 3 science records
        assert len(response.result.hits) <= 3

        # Every hit must be a science record — no history records should appear
        returned_ids = {hit.id for hit in response.result.hits}
        assert not returned_ids.intersection({"swf-hist-1", "swf-hist-2"}), (
            f"History records leaked through filter: {returned_ids}"
        )

        # All returned hits should have category=science in their fields
        for hit in response.result.hits:
            assert isinstance(hit, Hit)
            assert isinstance(hit.id, str)
            assert isinstance(hit.score, float)
            assert hit.id.startswith("swf-sci-"), f"Unexpected hit id: {hit.id!r}"
            if hit.fields:
                assert hit.fields.get("category") == "science", (
                    f"Expected category=science, got {hit.fields.get('category')!r}"
                )

        # Usage should be populated
        assert isinstance(response.usage, SearchUsage)
        assert response.usage.read_units > 0

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# search-with-filter — gRPC (delegates to REST)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_with_filter_grpc(client: Pinecone, api_key: str) -> None:
    """search() with a metadata filter via GrpcIndex (delegates to REST).

    Verifies that the filter parameter is correctly forwarded when using
    GrpcIndex.search() — GrpcIndex delegates search() to REST for integrated
    inference indexes.
    """
    name = unique_name("idx")
    namespace = "swf-ns"
    try:
        client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "text"},
                ),
            ),
        )

        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        index = client.index(name=name, grpc=True)

        index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "swf-sci-1", "text": "Quantum mechanics describes subatomic particles.", "category": "science"},
                {"_id": "swf-sci-2", "text": "DNA encodes the genetic information of organisms.", "category": "science"},
                {"_id": "swf-hist-1", "text": "The Roman Empire lasted for centuries.", "category": "history"},
            ],
        )

        # Wait for records to be searchable
        poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=3,
                inputs={"text": "science research"},
            ),
            check_fn=lambda r: len(r.result.hits) >= 2,
            timeout=120,
            description="records searchable (gRPC) before filter test",
        )

        # Search with filter via gRPC index (delegates to REST internally)
        response = index.search(
            namespace=namespace,
            top_k=5,
            inputs={"text": "science research"},
            filter={"category": {"$eq": "science"}},
        )

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) > 0

        # History records must not appear
        returned_ids = {hit.id for hit in response.result.hits}
        assert "swf-hist-1" not in returned_ids, (
            f"History record leaked through filter: {returned_ids}"
        )
        for hit in response.result.hits:
            assert isinstance(hit.id, str)
            assert isinstance(hit.score, float)

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_search_by_id_grpc(client: Pinecone, api_key: str) -> None:
    """search(id=...) via GrpcIndex (delegates to REST for integrated inference)."""
    name = unique_name("idx")
    namespace = "sid-ns"
    try:
        client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "text"},
                ),
            ),
        )

        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        index = client.index(name=name, grpc=True)

        index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "sid-1", "text": "Vector databases enable fast similarity search."},
                {"_id": "sid-2", "text": "RAG combines retrieval with language model generation."},
            ],
        )

        # Wait until records are searchable
        poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=2,
                inputs={"text": "similarity search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="records searchable (gRPC) before search-by-id",
        )

        # Search by ID via gRPC (delegates to REST under the hood)
        response = index.search(namespace=namespace, top_k=2, id="sid-1")

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) > 0
        for hit in response.result.hits:
            assert isinstance(hit.id, str)
            assert isinstance(hit.score, float)

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# search() input validation — REST sync (no real index needed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_input_validation_rest(client: Pinecone) -> None:
    """search() client-side validation raises PineconeValueError before any API call.

    Uses a fake host so no real index or network call is required; all checks
    fire synchronously before the HTTP request would be made.

    Verifies:
    - unified-vec-0047: empty or non-string namespace is rejected
    - unified-vec-0050: top_k < 1 is rejected
    - unified-vec-0051: rerank dict missing 'model' or 'rank_fields' is rejected
    """
    # Fake host — no describe-index call; validation fires before any HTTP request
    index = client.index(host="fake-index.svc.pinecone.io")

    # unified-vec-0047: non-string namespace (None) rejected
    with pytest.raises(PineconeValueError):
        index.search(namespace=None, top_k=1, inputs={"text": "hello"})  # type: ignore[arg-type]

    # unified-vec-0047: empty string namespace rejected
    with pytest.raises(PineconeValueError):
        index.search(namespace="", top_k=1, inputs={"text": "hello"})

    # unified-vec-0047: whitespace-only namespace rejected
    with pytest.raises(PineconeValueError):
        index.search(namespace="   ", top_k=1, inputs={"text": "hello"})

    # unified-vec-0050: top_k=0 rejected (must be >= 1)
    with pytest.raises(PineconeValueError):
        index.search(namespace="valid-ns", top_k=0, inputs={"text": "hello"})

    # unified-vec-0050: negative top_k rejected
    with pytest.raises(PineconeValueError):
        index.search(namespace="valid-ns", top_k=-1, inputs={"text": "hello"})

    # unified-vec-0051: rerank dict missing required 'model' key
    with pytest.raises(PineconeValueError):
        index.search(
            namespace="valid-ns",
            top_k=5,
            inputs={"text": "hello"},
            rerank={"rank_fields": ["text"]},
        )

    # unified-vec-0051: rerank dict missing required 'rank_fields' key
    with pytest.raises(PineconeValueError):
        index.search(
            namespace="valid-ns",
            top_k=5,
            inputs={"text": "hello"},
            rerank={"model": "bge-reranker-v2-m3"},
        )


# ---------------------------------------------------------------------------
# search_records() alias with typed SearchInputs + RerankConfig objects — REST
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_records_alias_with_typed_inputs_rest(client: Pinecone, api_key: str) -> None:
    """search_records() alias works with typed SearchInputs and RerankConfig objects.

    Verifies:
    - unified-vec-0031: search_records() is an alias for search() with identical behavior
    - Flexible input formats: SearchInputs(text=...) accepted in place of plain dict
    - Flexible input formats: RerankConfig(model=..., rank_fields=...) accepted in place of plain dict
    """
    name = unique_name("idx")
    namespace = "alias-ns"
    try:
        client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "text"},
                ),
            ),
        )

        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        index = client.index(name=name)

        index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "a-1", "text": "Vector databases enable fast similarity search at scale."},
                {"_id": "a-2", "text": "RAG combines retrieval with language model generation."},
                {"_id": "a-3", "text": "Embeddings are dense vector representations of text data."},
            ],
        )

        # Poll using plain dict to establish searchability before using typed objects
        poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=3,
                inputs={"text": "vector similarity search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="records searchable before alias test",
        )

        # Call search_records() alias with typed SearchInputs and RerankConfig objects
        response = index.search_records(
            namespace=namespace,
            top_k=3,
            inputs=SearchInputs(text="vector similarity search"),
            rerank=RerankConfig(model="bge-reranker-v2-m3", rank_fields=["text"], top_n=2),
        )

        assert isinstance(response, SearchRecordsResponse)
        assert isinstance(response.result, SearchResult)
        assert len(response.result.hits) > 0
        assert len(response.result.hits) <= 2  # top_n=2 caps hits after reranking

        for hit in response.result.hits:
            assert isinstance(hit, Hit)
            assert isinstance(hit.id, str)
            assert isinstance(hit.score, float)

        assert isinstance(response.usage, SearchUsage)
        assert response.usage.read_units > 0
        assert response.usage.rerank_units is not None
        assert response.usage.rerank_units > 0

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_search_records_alias_with_typed_inputs_grpc(client: Pinecone, api_key: str) -> None:
    """GrpcIndex.search_records() alias with typed SearchInputs delegates to REST search.

    Verifies that the gRPC index exposes the same search_records() alias and accepts
    typed SearchInputs in place of a plain dict.
    """
    name = unique_name("idx")
    namespace = "alias-grpc-ns"
    try:
        client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "text"},
                ),
            ),
        )
        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        index = client.index(name=name, grpc=True)

        index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "g-1", "text": "Vector databases enable fast similarity search."},
            ],
        )

        # Poll until searchable
        poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=1,
                inputs={"text": "vector search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="records searchable before grpc alias test",
        )

        # Call search_records() alias with typed SearchInputs
        response = index.search_records(
            namespace=namespace,
            top_k=1,
            inputs=SearchInputs(text="vector search"),
        )

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) > 0

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")
