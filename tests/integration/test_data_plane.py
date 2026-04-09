"""Integration tests for data-plane vector operations (sync REST + gRPC)."""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.responses import FetchResponse, ListItem, ListResponse, QueryResponse, UpsertResponse
from pinecone.models.vectors.vector import ScoredVector, Vector
from tests.integration.conftest import cleanup_resource, poll_until, unique_name

# ---------------------------------------------------------------------------
# delete-vectors — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_delete_vectors_rest(client: Pinecone) -> None:
    """Delete vectors by IDs via REST Index and verify they are gone."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        index.upsert(
            vectors=[
                {"id": "del-v1", "values": [0.1, 0.2]},
                {"id": "del-v2", "values": [0.3, 0.4]},
                {"id": "del-v3", "values": [0.5, 0.6]},
            ]
        )

        # Wait for all 3 vectors to be fetchable (eventual consistency)
        poll_until(
            query_fn=lambda: index.fetch(ids=["del-v1", "del-v2", "del-v3"]),
            check_fn=lambda r: len(r.vectors) == 3,
            timeout=120,
            description="all 3 vectors fetchable before delete",
        )

        # Delete just v1 and v2 by IDs
        result = index.delete(ids=["del-v1", "del-v2"])
        assert result is None  # delete returns None on success

        # Wait until deleted vectors are gone (eventual consistency)
        poll_until(
            query_fn=lambda: index.fetch(ids=["del-v1", "del-v2"]),
            check_fn=lambda r: len(r.vectors) == 0,
            timeout=120,
            description="deleted vectors gone after delete",
        )

        # Verify v3 is still present
        remaining = index.fetch(ids=["del-v3"])
        assert isinstance(remaining, FetchResponse)
        assert "del-v3" in remaining.vectors
        assert "del-v1" not in remaining.vectors
        assert "del-v2" not in remaining.vectors
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# delete-vectors — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0002: pinecone._grpc Rust extension not installed; ModuleNotFoundError on GrpcIndex creation",
)
def test_delete_vectors_grpc(client: Pinecone) -> None:
    """Delete vectors by IDs via GrpcIndex and verify they are gone."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        index.upsert(
            vectors=[
                {"id": "del-v1", "values": [0.1, 0.2]},
                {"id": "del-v2", "values": [0.3, 0.4]},
            ]
        )

        index.delete(ids=["del-v1"])

        remaining = index.fetch(ids=["del-v1", "del-v2"])
        assert "del-v1" not in remaining.vectors
        assert "del-v2" in remaining.vectors
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")

# ---------------------------------------------------------------------------
# upsert — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_vectors_rest(client: Pinecone) -> None:
    """Upsert vectors via REST Index and verify upserted_count."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        result = index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
                {"id": "v3", "values": [0.5, 0.6]},
            ]
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 3
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_query_by_vector_rest(client: Pinecone) -> None:
    """Query by vector via REST Index and verify matches structure."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
                {"id": "v3", "values": [0.5, 0.6]},
            ]
        )

        # Wait for all 3 vectors to be queryable (eventual consistency)
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=3),
            check_fn=lambda r: len(r.matches) == 3,
            timeout=120,
            description="all 3 vectors queryable after upsert",
        )

        result = index.query(vector=[0.1, 0.2], top_k=2)

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 2
        for match in result.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)
        # Scores must be in descending order
        scores = [m.score for m in result.matches]
        assert scores == sorted(scores, reverse=True)
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0002: pinecone._grpc Rust extension not installed; ModuleNotFoundError on GrpcIndex creation",
)
def test_upsert_vectors_grpc(client: Pinecone) -> None:
    """Upsert vectors via GrpcIndex and verify upserted_count."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        result = index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
                {"id": "v3", "values": [0.5, 0.6]},
            ]
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 3
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0002: pinecone._grpc Rust extension not installed; ModuleNotFoundError on GrpcIndex creation",
)
def test_query_by_vector_grpc(client: Pinecone) -> None:
    """Query by vector via GrpcIndex and verify matches structure."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
            ]
        )

        result = index.query(vector=[0.1, 0.2], top_k=1)

        assert isinstance(result, QueryResponse)
        assert len(result.matches) >= 1
        assert result.matches[0].id == "v1"
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# fetch — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_fetch_vectors_rest(client: Pinecone) -> None:
    """Fetch vectors by ID via REST Index and verify returned vector data."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
                {"id": "v3", "values": [0.5, 0.6]},
            ]
        )

        # Wait for vectors to be fetchable (eventual consistency)
        poll_until(
            query_fn=lambda: index.fetch(ids=["v1", "v2", "v3"]),
            check_fn=lambda r: len(r.vectors) == 3,
            timeout=120,
            description="all 3 vectors fetchable after upsert",
        )

        result = index.fetch(ids=["v1", "v2"])

        assert isinstance(result, FetchResponse)
        assert "v1" in result.vectors
        assert "v2" in result.vectors
        assert "v3" not in result.vectors

        v1 = result.vectors["v1"]
        assert isinstance(v1, Vector)
        assert v1.id == "v1"
        assert len(v1.values) == 2
        assert abs(v1.values[0] - 0.1) < 1e-5
        assert abs(v1.values[1] - 0.2) < 1e-5

        assert isinstance(result.namespace, str)
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# fetch — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0002: pinecone._grpc Rust extension not installed; ModuleNotFoundError on GrpcIndex creation",
)
def test_fetch_vectors_grpc(client: Pinecone) -> None:
    """Fetch vectors by ID via GrpcIndex and verify returned vector data."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
            ]
        )

        result = index.fetch(ids=["v1", "v2"])

        assert isinstance(result, FetchResponse)
        assert "v1" in result.vectors
        assert "v2" in result.vectors
        assert isinstance(result.vectors["v1"], Vector)
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# list-vectors — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_list_vectors_rest(client: Pinecone) -> None:
    """List vectors via REST Index and verify pagination structure and IDs."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        index.upsert(
            vectors=[
                {"id": "lst-v1", "values": [0.1, 0.2]},
                {"id": "lst-v2", "values": [0.3, 0.4]},
                {"id": "lst-v3", "values": [0.5, 0.6]},
            ]
        )

        # Wait for all 3 vectors to appear in list results (eventual consistency)
        def _collect_ids() -> list[str]:
            ids: list[str] = []
            for page in index.list(prefix="lst-"):
                for item in page.vectors:
                    if item.id is not None:
                        ids.append(item.id)
            return ids

        poll_until(
            query_fn=_collect_ids,
            check_fn=lambda ids: len(ids) >= 3,
            timeout=120,
            description="all 3 vectors listable after upsert",
        )

        # Collect all pages and verify structure
        pages: list[ListResponse] = []
        for page in index.list(prefix="lst-"):
            pages.append(page)

        assert len(pages) >= 1, "expected at least one page"
        all_ids: list[str] = []
        for page in pages:
            assert isinstance(page, ListResponse)
            assert isinstance(page.namespace, str)
            for item in page.vectors:
                assert isinstance(item, ListItem)
                assert isinstance(item.id, str)
                all_ids.append(item.id)

        assert "lst-v1" in all_ids
        assert "lst-v2" in all_ids
        assert "lst-v3" in all_ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# list-vectors — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0002: pinecone._grpc Rust extension not installed; ModuleNotFoundError on GrpcIndex creation",
)
def test_list_vectors_grpc(client: Pinecone) -> None:
    """List vectors via GrpcIndex and verify structure."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        index.upsert(
            vectors=[
                {"id": "lst-v1", "values": [0.1, 0.2]},
                {"id": "lst-v2", "values": [0.3, 0.4]},
            ]
        )

        all_ids: list[str] = []
        for page in index.list(prefix="lst-"):
            assert isinstance(page, ListResponse)
            for item in page.vectors:
                assert isinstance(item, ListItem)
                if item.id is not None:
                    all_ids.append(item.id)

        assert "lst-v1" in all_ids
        assert "lst-v2" in all_ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")
