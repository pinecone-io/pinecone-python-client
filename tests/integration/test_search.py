"""Integration tests for search-records with integrated inference — sync (REST + gRPC)."""

from __future__ import annotations

import pytest

from pinecone import EmbedConfig, IntegratedSpec, Pinecone
from pinecone.models.vectors.search import Hit, SearchRecordsResponse, SearchResult, SearchUsage
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
# search-records — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason=(
        "SDK bug IT-0002: pinecone._grpc Rust extension not installed; "
        "ModuleNotFoundError on GrpcIndex creation"
    ),
)
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
