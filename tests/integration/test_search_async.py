"""Integration tests for search-records with integrated inference — async (REST) transport.

Note on integrated index creation
----------------------------------
The SDK's ``async_client.indexes.create()`` with ``IntegratedSpec`` incorrectly POSTs to
``/indexes`` instead of the correct ``/indexes/create-for-model`` endpoint (IT-0003).
To test the ``search()`` method independently of that bug, these tests create the
integrated index directly via ``httpx`` — an explicit workaround that will be removed
once IT-0003 is fixed.  All other operations (upsert_records, search, delete) use
the SDK exclusively.
"""

from __future__ import annotations

import httpx
import pytest

from pinecone import AsyncPinecone, Pinecone
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
