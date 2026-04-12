"""Integration tests for search with integrated inference — async (REST) transport.

Covers:
  - search-records: basic text search (async)
  - search-with-rerank: text search with inline reranking (async)
  - search-by-id: search using a stored record ID as the query vector (async)
  - search-with-filter: text search with a metadata filter expression (async)

Note on integrated index creation
----------------------------------
The SDK's ``async_client.indexes.create()`` with ``IntegratedSpec`` incorrectly POSTs to
``/indexes`` instead of the correct ``/indexes/create-for-model`` endpoint (IT-0003).
To test search methods independently of that bug, these tests create the integrated index
directly via ``httpx`` — an explicit workaround that will be removed once IT-0003 is fixed.
All other operations (upsert_records, search, delete) use the SDK exclusively.
"""

from __future__ import annotations

import httpx
import pytest

from pinecone import AsyncPinecone, Pinecone
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.vectors.search import Hit, SearchRecordsResponse, SearchResult, SearchUsage
from tests.integration.conftest import (
    async_cleanup_resource,
    async_poll_until,
    unique_name,
    wait_for_ready,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_VERSION = "2025-10"
_BASE_URL = "https://api.pinecone.io"


def _create_integrated_index(api_key: str, name: str) -> None:
    """Create an integrated index via the correct endpoint (workaround for IT-0003)."""
    headers = {
        "Api-Key": api_key,
        "X-Pinecone-API-Version": _API_VERSION,
        "Content-Type": "application/json",
    }
    body = {
        "name": name,
        "cloud": "aws",
        "region": "us-east-1",
        "embed": {
            "model": "multilingual-e5-large",
            "field_map": {"text": "text"},
        },
    }
    with httpx.Client(timeout=30) as http:
        response = http.post(
            f"{_BASE_URL}/indexes/create-for-model",
            headers=headers,
            json=body,
        )
        response.raise_for_status()


# ---------------------------------------------------------------------------
# search-records — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_records_rest_async(
    async_client: AsyncPinecone, client: Pinecone, api_key: str
) -> None:
    """search() with text inputs on an integrated index returns SearchRecordsResponse (async)."""
    name = unique_name("idx")
    namespace = "srch-ns"
    try:
        # Create integrated index via direct HTTP call (SDK uses wrong endpoint — IT-0003)
        _create_integrated_index(api_key, name)

        # Wait for the index to become ready using the sync client (async describe can't be
        # called from sync wait_for_ready)
        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        # Populate the async client's host cache before calling index()
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        # Upsert records: text fields are embedded server-side
        await index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "doc-1", "text": "Vector databases enable fast similarity search."},
                {"_id": "doc-2", "text": "RAG combines retrieval with language model generation."},
                {"_id": "doc-3", "text": "Embeddings are dense vector representations of data."},
            ],
        )

        # Wait for records to be searchable (eventual consistency)
        response = await async_poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=3,
                inputs={"text": "similarity search with embeddings"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="records searchable after upsert (async)",
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
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# search-with-rerank — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_with_rerank_rest_async(
    async_client: AsyncPinecone, client: Pinecone, api_key: str
) -> None:
    """search() with inline rerank parameter re-ranks hits and populates usage.rerank_units (async)."""
    name = unique_name("idx")
    namespace = "rerank-ns"
    try:
        # Create integrated index via direct HTTP call (SDK uses wrong endpoint — IT-0003)
        _create_integrated_index(api_key, name)

        # Wait for the index to become ready using the sync client
        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        # Populate the async client's host cache before calling index()
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        # Upsert records with varied text content
        await index.upsert_records(
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
        await async_poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=5,
                inputs={"text": "vector database similarity search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="records searchable before rerank test (async)",
        )

        # Search with inline reranking — top_n=3 limits results after reranking
        response = await index.search(
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
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# search-by-id — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_by_id_rest_async(
    async_client: AsyncPinecone, client: Pinecone, api_key: str
) -> None:
    """search(id=...) uses a stored record's embedding as the query vector (async).

    Verifies that search-by-id returns a SearchRecordsResponse with the same
    structure as search-by-text: hits list, hit.id (str), hit.score (float),
    and usage.read_units > 0.
    """
    name = unique_name("idx")
    namespace = "sid-ns"
    try:
        # Create integrated index via direct HTTP call (SDK uses wrong endpoint — IT-0003)
        _create_integrated_index(api_key, name)

        # Wait for the index to become ready using the sync client
        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        # Populate the async client's host cache before calling index()
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        # Upsert records — embeddings are generated server-side from the text field
        await index.upsert_records(
            namespace=namespace,
            records=[
                {"_id": "sid-1", "text": "Vector databases enable fast similarity search."},
                {"_id": "sid-2", "text": "RAG combines retrieval with language model generation."},
                {"_id": "sid-3", "text": "Embeddings are dense vector representations of data."},
            ],
        )

        # Wait until records are searchable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=3,
                inputs={"text": "similarity search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="records searchable (async) before search-by-id",
        )

        # Search using an existing record ID as the query seed
        response = await index.search(namespace=namespace, top_k=3, id="sid-1")

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
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# search-with-filter — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_with_filter_rest_async(
    async_client: AsyncPinecone, client: Pinecone, api_key: str
) -> None:
    """search() with a metadata filter returns only records matching the filter (async).

    Upserts records with two distinct category values ('science' and 'history').
    A filter for category=science must return only science records; no history
    record should appear. The fields parameter is used to verify category values
    in the returned hits.
    """
    name = unique_name("idx")
    namespace = "swf-ns"
    try:
        # Create integrated index via direct HTTP call (SDK uses wrong endpoint — IT-0003)
        _create_integrated_index(api_key, name)

        # Wait for the index to become ready using the sync client
        wait_for_ready(
            lambda: client.indexes.describe(name).status.ready,
            timeout=300,
            description=f"integrated index {name!r}",
        )

        # Populate the async client's host cache before calling index()
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        # Upsert records with a 'category' metadata field for filtering
        await index.upsert_records(
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
        await async_poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=5,
                inputs={"text": "science and research"},
            ),
            check_fn=lambda r: len(r.result.hits) >= 3,
            timeout=120,
            description="all records searchable before filter test (async)",
        )

        # Search with category=science filter, requesting category field in hits
        response = await index.search(
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
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# search() input validation — REST async (no real index needed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_input_validation_rest_async(async_client: AsyncPinecone) -> None:
    """search() client-side validation raises PineconeValueError before any API call (async).

    Uses a fake host so no real index or network call is required; all checks
    fire synchronously within the async function before any await.

    Verifies:
    - unified-vec-0047: empty or non-string namespace is rejected
    - unified-vec-0050: top_k < 1 is rejected
    - unified-vec-0051: rerank dict missing 'model' or 'rank_fields' is rejected
    """
    # Fake host — no describe-index call; validation fires before any HTTP request
    index = async_client.index(host="fake-index.svc.pinecone.io")
    try:
        # unified-vec-0047: non-string namespace (None) rejected
        with pytest.raises(PineconeValueError):
            await index.search(namespace=None, top_k=1, inputs={"text": "hello"})  # type: ignore[arg-type]

        # unified-vec-0047: empty string namespace rejected
        with pytest.raises(PineconeValueError):
            await index.search(namespace="", top_k=1, inputs={"text": "hello"})

        # unified-vec-0047: whitespace-only namespace rejected
        with pytest.raises(PineconeValueError):
            await index.search(namespace="   ", top_k=1, inputs={"text": "hello"})

        # unified-vec-0050: top_k=0 rejected (must be >= 1)
        with pytest.raises(PineconeValueError):
            await index.search(namespace="valid-ns", top_k=0, inputs={"text": "hello"})

        # unified-vec-0050: negative top_k rejected
        with pytest.raises(PineconeValueError):
            await index.search(namespace="valid-ns", top_k=-1, inputs={"text": "hello"})

        # unified-vec-0051: rerank dict missing required 'model' key
        with pytest.raises(PineconeValueError):
            await index.search(
                namespace="valid-ns",
                top_k=5,
                inputs={"text": "hello"},
                rerank={"rank_fields": ["text"]},
            )

        # unified-vec-0051: rerank dict missing required 'rank_fields' key
        with pytest.raises(PineconeValueError):
            await index.search(
                namespace="valid-ns",
                top_k=5,
                inputs={"text": "hello"},
                rerank={"model": "bge-reranker-v2-m3"},
            )
    finally:
        await index.close()
