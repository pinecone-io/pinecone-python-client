"""Integration tests for data-plane vector operations (sync REST + gRPC)."""

from __future__ import annotations

from concurrent.futures import as_completed

import pytest

from pinecone import GrpcIndex, Index, Pinecone
from pinecone.errors import ApiError, ConflictError, PineconeValueError
from pinecone.grpc.future import PineconeFuture
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.namespaces.models import ListNamespacesResponse, NamespaceDescription
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListItem,
    ListResponse,
    NamespaceSummary,
    QueryResponse,
    UpdateResponse,
    UpsertResponse,
)
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
# delete-vectors — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
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
# upsert — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
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
# query — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
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
# fetch — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
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
            return [
                item.id
                for page in index.list(prefix="lst-")
                for item in page.vectors
                if item.id is not None
            ]

        poll_until(
            query_fn=_collect_ids,
            check_fn=lambda ids: len(ids) >= 3,
            timeout=120,
            description="all 3 vectors listable after upsert",
        )

        # Collect all pages and verify structure
        pages: list[ListResponse] = list(index.list(prefix="lst-"))

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
# list-vectors — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
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


# ---------------------------------------------------------------------------
# update-vectors — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_update_vectors_rest(client: Pinecone) -> None:
    """Update a vector's values via REST Index and verify the change is reflected."""
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
                {"id": "upd-v1", "values": [0.1, 0.2]},
                {"id": "upd-v2", "values": [0.3, 0.4]},
            ]
        )

        # Wait for vectors to be fetchable (eventual consistency)
        poll_until(
            query_fn=lambda: index.fetch(ids=["upd-v1", "upd-v2"]),
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="vectors fetchable before update",
        )

        # Update upd-v1 with new values
        result = index.update(id="upd-v1", values=[0.9, 0.8])

        assert isinstance(result, UpdateResponse)
        # The update API returns {} on success; matched_records may be None
        assert result.matched_records is None or isinstance(result.matched_records, int)

        # Poll until the updated values are reflected
        poll_until(
            query_fn=lambda: index.fetch(ids=["upd-v1"]),
            check_fn=lambda r: (
                "upd-v1" in r.vectors
                and len(r.vectors["upd-v1"].values) == 2
                and abs(r.vectors["upd-v1"].values[0] - 0.9) < 1e-4
            ),
            timeout=120,
            description="updated values reflected in fetch",
        )

        # Verify upd-v2 was not modified
        check = index.fetch(ids=["upd-v2"])
        assert abs(check.vectors["upd-v2"].values[0] - 0.3) < 1e-4
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# update-vectors — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_update_vectors_grpc(client: Pinecone) -> None:
    """Update a vector's values via GrpcIndex and verify the change is reflected."""
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
                {"id": "upd-v1", "values": [0.1, 0.2]},
            ]
        )

        poll_until(
            query_fn=lambda: index.fetch(ids=["upd-v1"]),
            check_fn=lambda r: "upd-v1" in r.vectors,
            timeout=120,
            description="vector fetchable before update",
        )

        result = index.update(id="upd-v1", values=[0.9, 0.8])
        assert isinstance(result, UpdateResponse)

        poll_until(
            query_fn=lambda: index.fetch(ids=["upd-v1"]),
            check_fn=lambda r: (
                "upd-v1" in r.vectors and abs(r.vectors["upd-v1"].values[0] - 0.9) < 1e-4
            ),
            timeout=120,
            description="updated values reflected in fetch",
        )
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# describe-stats — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_index_stats_rest(client: Pinecone) -> None:
    """Call describe_index_stats() via REST Index and verify the response structure."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=3,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        # Upsert a few vectors so stats are non-trivial
        index.upsert(
            vectors=[
                {"id": "st-v1", "values": [0.1, 0.2, 0.3]},
                {"id": "st-v2", "values": [0.4, 0.5, 0.6]},
            ]
        )

        # Wait until at least 1 vector is counted in stats (eventual consistency)
        poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 1,
            timeout=120,
            description="at least 1 vector counted in stats after upsert",
        )

        stats = index.describe_index_stats()

        assert isinstance(stats, DescribeIndexStatsResponse)
        assert stats.dimension == 3
        assert stats.total_vector_count >= 1
        assert isinstance(stats.index_fullness, float)
        assert 0.0 <= stats.index_fullness <= 1.0
        assert isinstance(stats.namespaces, dict)
        assert len(stats.namespaces) >= 1
        # Verify at least one namespace entry has the expected structure
        for ns_name, ns in stats.namespaces.items():
            assert isinstance(ns_name, str)
            assert isinstance(ns, NamespaceSummary)
            assert isinstance(ns.vector_count, int)
            assert ns.vector_count >= 0
        # Total across namespaces should match total_vector_count
        ns_total = sum(ns.vector_count for ns in stats.namespaces.values())
        assert ns_total == stats.total_vector_count
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# describe-stats — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_index_stats_grpc(client: Pinecone) -> None:
    """Call describe_index_stats() via GrpcIndex and verify response structure."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=3,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        index.upsert(
            vectors=[
                {"id": "st-v1", "values": [0.1, 0.2, 0.3]},
            ]
        )

        stats = index.describe_index_stats()

        assert isinstance(stats, DescribeIndexStatsResponse)
        assert stats.dimension == 3
        assert isinstance(stats.total_vector_count, int)
        assert isinstance(stats.index_fullness, float)
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# namespaces — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespaces_rest(client: Pinecone) -> None:
    """Upsert to named namespace, query within it, verify namespace isolation."""
    name = unique_name("idx")
    named_ns = "ns-alpha"
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        # Upsert vectors into named namespace
        ns_result = index.upsert(
            vectors=[
                {"id": "ns-v1", "values": [0.1, 0.2]},
                {"id": "ns-v2", "values": [0.3, 0.4]},
            ],
            namespace=named_ns,
        )
        assert isinstance(ns_result, UpsertResponse)
        assert ns_result.upserted_count == 2

        # Upsert different vectors into the default namespace
        def_result = index.upsert(
            vectors=[
                {"id": "def-v1", "values": [0.9, 0.8]},
            ],
            namespace="",
        )
        assert isinstance(def_result, UpsertResponse)
        assert def_result.upserted_count == 1

        # Wait until ns-alpha vectors are queryable in the named namespace
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="named namespace vectors queryable",
        )

        # Query in the named namespace
        ns_query = index.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns)
        assert isinstance(ns_query, QueryResponse)
        assert ns_query.namespace == named_ns
        ns_ids = {m.id for m in ns_query.matches}
        assert "ns-v1" in ns_ids
        assert "ns-v2" in ns_ids
        # Default namespace vectors must NOT appear in named-namespace query
        assert "def-v1" not in ns_ids

        # Verify stats shows both namespaces
        poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda s: s.total_vector_count >= 3,
            timeout=120,
            description="stats reflect all 3 vectors",
        )
        stats = index.describe_index_stats()
        assert isinstance(stats.namespaces, dict)
        assert named_ns in stats.namespaces
        assert stats.namespaces[named_ns].vector_count == 2
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# namespaces — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespaces_grpc(client: Pinecone) -> None:
    """Upsert to named namespace via GrpcIndex and query within it."""
    name = unique_name("idx")
    named_ns = "ns-alpha"
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
                {"id": "ns-v1", "values": [0.1, 0.2]},
                {"id": "ns-v2", "values": [0.3, 0.4]},
            ],
            namespace=named_ns,
        )

        result = index.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns)
        assert isinstance(result, QueryResponse)
        assert result.namespace == named_ns
        ids = {m.id for m in result.matches}
        assert "ns-v1" in ids or "ns-v2" in ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# list_paginated — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_paginated_returns_single_page_rest(client: Pinecone) -> None:
    """list_paginated() returns one page with correct structure; no token on last page."""
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

        # Upsert 4 vectors with a shared prefix so we can filter and verify limit
        index.upsert(
            vectors=[
                {"id": "pg-v1", "values": [0.1, 0.2]},
                {"id": "pg-v2", "values": [0.3, 0.4]},
                {"id": "pg-v3", "values": [0.5, 0.6]},
                {"id": "pg-v4", "values": [0.7, 0.8]},
            ]
        )

        # Wait until all 4 vectors appear in list results
        poll_until(
            query_fn=lambda: index.list_paginated(prefix="pg-", limit=100),
            check_fn=lambda r: len(r.vectors) >= 4,
            timeout=120,
            description="all 4 vectors listable after upsert",
        )

        # 1. list_paginated() returns a ListResponse (not a generator)
        page = index.list_paginated(prefix="pg-", limit=100)
        assert isinstance(page, ListResponse)

        # 2. Vector list contains ListItem objects with string IDs
        assert isinstance(page.vectors, list)
        assert len(page.vectors) >= 4
        for item in page.vectors:
            assert isinstance(item, ListItem)
            assert isinstance(item.id, str)

        # 3. All 4 IDs are present
        ids = {item.id for item in page.vectors}
        assert {"pg-v1", "pg-v2", "pg-v3", "pg-v4"} <= ids

        # 4. Namespace field is a string
        assert isinstance(page.namespace, str)

        # 5. No pagination token when the page contains all results (last page)
        #    The claim unified-vec-0056 says: no token when this is the final page.
        assert page.pagination is None or page.pagination.next is None

        # 6. list_paginated with limit=2 returns at most 2 items (limit is respected)
        limited_page = index.list_paginated(prefix="pg-", limit=2)
        assert isinstance(limited_page, ListResponse)
        assert len(limited_page.vectors) <= 2
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# list_paginated — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_paginated_returns_single_page_grpc(client: Pinecone) -> None:
    """list_paginated() via GrpcIndex returns one page with correct structure."""
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
                {"id": "pg-v1", "values": [0.1, 0.2]},
                {"id": "pg-v2", "values": [0.3, 0.4]},
                {"id": "pg-v3", "values": [0.5, 0.6]},
            ]
        )

        # Wait until all 3 vectors are listable
        poll_until(
            query_fn=lambda: index.list_paginated(prefix="pg-", limit=100),
            check_fn=lambda r: len(r.vectors) >= 3,
            timeout=120,
            description="all 3 vectors listable after upsert",
        )

        page = index.list_paginated(prefix="pg-", limit=100)
        assert isinstance(page, ListResponse)
        assert isinstance(page.vectors, list)
        assert len(page.vectors) >= 3
        for item in page.vectors:
            assert isinstance(item, ListItem)
            assert isinstance(item.id, str)
        ids = {item.id for item in page.vectors}
        assert {"pg-v1", "pg-v2", "pg-v3"} <= ids
        # No pagination token on the final page
        assert page.pagination is None or page.pagination.next is None
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# describe-stats with filter — serverless rejects (REST sync)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_index_stats_filter_unsupported_on_serverless_rest(client: Pinecone) -> None:
    """Verify describe_index_stats(filter=...) raises ApiError(400) on a serverless index."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=3,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        # Upsert one vector so the index has some content
        index.upsert(vectors=[{"id": "fs-v1", "values": [0.1, 0.2, 0.3]}])

        # The filter parameter is not supported on serverless/starter indexes —
        # the API returns 400 and the SDK should surface it as ApiError.
        with pytest.raises(ApiError) as exc_info:
            index.describe_index_stats(filter={"tag": {"$eq": "a"}})

        assert exc_info.value.status_code == 400
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# describe-stats with filter — serverless rejects (gRPC)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_index_stats_filter_unsupported_on_serverless_grpc(client: Pinecone) -> None:
    """Verify describe_index_stats(filter=...) raises an error on a serverless index via gRPC."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=3,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        index.upsert(vectors=[{"id": "fg-v1", "values": [0.1, 0.2, 0.3]}])

        # The filter parameter is not supported on serverless/starter indexes via gRPC either.
        with pytest.raises(Exception):
            index.describe_index_stats(filter={"tag": {"$eq": "a"}})
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# namespace CRUD — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespace_crud_lifecycle_rest(client: Pinecone) -> None:
    """create_namespace / describe_namespace / list_namespaces_paginated / delete_namespace round-trip.

    Verifies claims:
    - unified-ns-0001: Can create a named namespace.
    - unified-ns-0002: Creation returns name and record_count == 0.
    - unified-ns-0003: Can describe a namespace by name.
    - unified-ns-0004: Can delete a namespace by name.
    - unified-ns-0005: Can list all namespaces with optional prefix filtering.
    - unified-ns-0008: Namespace list response omits pagination token on the final page.
    """
    name = unique_name("idx")
    ns_name = "crud-ns-alpha"
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        # 1. Create namespace — returns NamespaceDescription with record_count == 0
        created = index.create_namespace(name=ns_name)
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name
        assert created.record_count == 0  # unified-ns-0002

        # 2. Describe namespace — returns same details
        described = index.describe_namespace(name=ns_name)
        assert isinstance(described, NamespaceDescription)
        assert described.name == ns_name
        assert isinstance(described.record_count, int)

        # 3. Namespace appears in list_namespaces_paginated with prefix match
        list_resp = index.list_namespaces_paginated(prefix="crud-ns-", limit=100)
        assert isinstance(list_resp, ListNamespacesResponse)
        ns_names = [ns.name for ns in list_resp.namespaces]
        assert ns_name in ns_names

        # Each entry is a NamespaceDescription with a string name
        for ns in list_resp.namespaces:
            assert isinstance(ns, NamespaceDescription)
            assert isinstance(ns.name, str)
            assert isinstance(ns.record_count, int)

        # 4. Pagination token is absent on the final page (unified-ns-0008)
        assert list_resp.pagination is None or list_resp.pagination.next is None

        # 5. Delete namespace — returns None on success
        result = index.delete_namespace(name=ns_name)
        assert result is None  # unified-ns-0004

        # 6. After deletion, namespace no longer appears in listing
        post_delete = index.list_namespaces_paginated(prefix="crud-ns-", limit=100)
        assert isinstance(post_delete, ListNamespacesResponse)
        post_names = [ns.name for ns in post_delete.namespaces]
        assert ns_name not in post_names
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# list_namespaces generator — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_namespaces_generator_rest(client: Pinecone) -> None:
    """list_namespaces() generator yields ListNamespacesResponse pages with NamespaceDescription
    items; generator follows pagination tokens automatically until exhausted.

    Verifies claims:
    - unified-ns-0007: The namespace list generator yields individual NamespaceDescription
      objects per page; the paginated method returns one page (not auto-paginating).
    - unified-ns-0005: Can list all namespaces in an index with optional prefix filtering.
    """
    name = unique_name("idx")
    ns_a = "lnsgen-ns-a"
    ns_b = "lnsgen-ns-b"
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        # Upsert one vector into each namespace to create them implicitly
        index.upsert(vectors=[{"id": "lnsg-v1", "values": [0.1, 0.2]}], namespace=ns_a)
        index.upsert(vectors=[{"id": "lnsg-v2", "values": [0.3, 0.4]}], namespace=ns_b)

        # Poll until both namespaces appear in list_namespaces_paginated
        poll_until(
            query_fn=lambda: index.list_namespaces_paginated(prefix="lnsgen-ns-", limit=100),
            check_fn=lambda r: len(r.namespaces) >= 2,
            timeout=120,
            description="both lnsgen-ns-* namespaces appear via list_namespaces_paginated",
        )

        # --- Exercise the generator ---
        pages = list(index.list_namespaces(prefix="lnsgen-ns-"))
        assert len(pages) >= 1, "list_namespaces() generator must yield at least one page"

        # Collect all namespace names across all yielded pages
        all_ns_names = [ns.name for page in pages for ns in page.namespaces]
        assert ns_a in all_ns_names, f"Expected {ns_a!r} in generator output; got {all_ns_names}"
        assert ns_b in all_ns_names, f"Expected {ns_b!r} in generator output; got {all_ns_names}"

        # Verify shape of every yielded page and its namespace descriptions
        for page in pages:
            assert isinstance(page, ListNamespacesResponse)
            assert len(page.namespaces) >= 1, (
                "Each yielded page must contain at least one namespace"
            )
            for ns in page.namespaces:
                assert isinstance(ns, NamespaceDescription)
                assert isinstance(ns.name, str) and ns.name.startswith("lnsgen-ns-")
                assert isinstance(ns.record_count, int) and ns.record_count >= 0
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# list_paginated multi-page — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_paginated_multi_page_rest(client: Pinecone) -> None:
    """list_paginated() with limit=2 returns a token when more pages exist; following the token
    reaches the next page; the final page has no token.

    Verifies claims:
    - unified-vec-0030: paginated list method returns a single page (caller must follow the token
      explicitly — it does not auto-paginate)
    - unified-vec-0056: list-paginated returns no pagination token when the current page is the last
    - unified-pag-0002: vector listing supports cursor-based pagination via a single-page method
    """
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

        # Upsert 4 vectors with a shared prefix
        index.upsert(
            vectors=[
                {"id": "mp-v1", "values": [0.1, 0.2]},
                {"id": "mp-v2", "values": [0.3, 0.4]},
                {"id": "mp-v3", "values": [0.5, 0.6]},
                {"id": "mp-v4", "values": [0.7, 0.8]},
            ]
        )

        # Wait until all 4 vectors appear in list results
        poll_until(
            query_fn=lambda: index.list_paginated(prefix="mp-", limit=100),
            check_fn=lambda r: len(r.vectors) >= 4,
            timeout=120,
            description="all 4 vectors listable after upsert",
        )

        # Traverse all pages manually using limit=2 (forces at least 2 pages)
        all_ids: list[str] = []
        token: str | None = None
        pages_seen = 0

        while True:
            page = index.list_paginated(prefix="mp-", limit=2, pagination_token=token)
            assert isinstance(page, ListResponse)
            assert isinstance(page.vectors, list)
            assert len(page.vectors) <= 2  # limit is respected each page

            for item in page.vectors:
                assert isinstance(item, ListItem)
                assert isinstance(item.id, str)
                all_ids.append(item.id)

            pages_seen += 1

            # Extract next token (if any)
            if page.pagination is not None and page.pagination.next is not None:
                # More pages exist — token should be a non-empty string
                assert isinstance(page.pagination.next, str)
                assert len(page.pagination.next) > 0
                token = page.pagination.next
            else:
                # Final page — no token (unified-vec-0056)
                break

        # Collected all 4 IDs across pages
        assert {"mp-v1", "mp-v2", "mp-v3", "mp-v4"} <= set(all_ids)
        # Required at least 2 pages (limit=2, 4 vectors → ≥2 pages)
        assert pages_seen >= 2
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# list-paginated — multi-page (gRPC)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_paginated_multi_page_grpc(client: Pinecone) -> None:
    """list_paginated() via GrpcIndex with limit=2 spans multiple pages; pagination tokens
    are returned when more pages exist; the final page has no token.

    Verifies claims (gRPC transport parity with REST):
    - unified-vec-0030: paginated list returns a single page; caller must follow the
      token explicitly — it does not auto-paginate
    - unified-vec-0056: list-paginated returns no pagination token when the current
      page is the last one
    - unified-pag-0002: vector listing supports cursor-based pagination via a
      single-page method

    This test is the gRPC counterpart of test_list_paginated_multi_page_rest.
    The single-page gRPC path is covered by test_list_paginated_returns_single_page_grpc;
    this test covers the multi-page pagination token flow on gRPC.
    """
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

        # Upsert 4 vectors with a shared prefix so limit=2 forces ≥ 2 pages
        index.upsert(
            vectors=[
                {"id": "gmp-v1", "values": [0.1, 0.2]},
                {"id": "gmp-v2", "values": [0.3, 0.4]},
                {"id": "gmp-v3", "values": [0.5, 0.6]},
                {"id": "gmp-v4", "values": [0.7, 0.8]},
            ]
        )

        # Wait until all 4 vectors appear in list results (eventual consistency)
        poll_until(
            query_fn=lambda: index.list_paginated(prefix="gmp-", limit=100),
            check_fn=lambda r: len(r.vectors) >= 4,
            timeout=120,
            description="all 4 gmp- vectors listable after upsert (gRPC)",
        )

        # Traverse all pages manually using limit=2 — forces at least 2 pages
        all_ids: list[str] = []
        token: str | None = None
        pages_seen = 0

        while True:
            page = index.list_paginated(prefix="gmp-", limit=2, pagination_token=token)
            assert isinstance(page, ListResponse)
            assert isinstance(page.vectors, list)
            # Limit must be respected on every page
            assert len(page.vectors) <= 2, (
                f"Page {pages_seen} returned {len(page.vectors)} vectors (limit=2)"
            )

            for item in page.vectors:
                assert isinstance(item, ListItem)
                assert isinstance(item.id, str)
            all_ids.extend(item.id for item in page.vectors)

            pages_seen += 1

            if page.pagination is not None and page.pagination.next is not None:
                # More pages available — token must be a non-empty string
                assert isinstance(page.pagination.next, str)
                assert len(page.pagination.next) > 0
                token = page.pagination.next
            else:
                # Final page — pagination token must be absent (unified-vec-0056)
                break

        # All 4 vectors must be collected across the pages
        assert {"gmp-v1", "gmp-v2", "gmp-v3", "gmp-v4"} <= set(all_ids)
        # Must have traversed at least 2 pages (4 vectors ÷ limit=2)
        assert pages_seen >= 2
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# delete-nonexistent-ids — REST sync + gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_nonexistent_ids_returns_none_rest(client: Pinecone) -> None:
    """Delete with IDs that don't exist in the namespace returns None without error.

    Verifies unified-vec-0032: "Deleting vectors does not raise an error when
    the specified IDs do not exist."

    Two sub-cases:
    1. IDs never upserted — tested after namespace is established (fresh namespace
       raises 404; the claim applies within an existing namespace).
    2. ID that was upserted, deleted, then deleted again (idempotency).
    """
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

        # Establish the default namespace by upserting a sentinel vector
        index.upsert(vectors=[{"id": "dn-v1", "values": [0.1, 0.9]}])
        poll_until(
            query_fn=lambda: index.fetch(ids=["dn-v1"]),
            check_fn=lambda r: len(r.vectors) == 1,
            timeout=120,
            description="dn-v1 fetchable (namespace established)",
        )

        # Sub-case 1: delete IDs that were never upserted — should return None, not raise
        result = index.delete(ids=["never-existed-x", "never-existed-y"])
        assert result is None

        # Sub-case 2: delete dn-v1 (exists), then delete it again (already gone)
        first = index.delete(ids=["dn-v1"])
        assert first is None

        # Second delete — vector is gone; must still return None (idempotency)
        second = index.delete(ids=["dn-v1"])
        assert second is None

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_delete_nonexistent_ids_returns_none_grpc(client: Pinecone) -> None:
    """Same as REST variant but over gRPC transport (unified-vec-0032)."""
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

        # Establish namespace by upserting a sentinel vector
        index.upsert(vectors=[{"id": "dn-g1", "values": [0.2, 0.8]}])
        poll_until(
            query_fn=lambda: index.fetch(ids=["dn-g1"]),
            check_fn=lambda r: len(r.vectors) == 1,
            timeout=120,
            description="dn-g1 fetchable (namespace established for gRPC test)",
        )

        # Sub-case 1: delete IDs that were never upserted
        result = index.delete(ids=["never-existed-grpc-x", "never-existed-grpc-y"])
        assert result is None

        # Sub-case 2: delete dn-g1 (exists), then delete again (already gone)
        first = index.delete(ids=["dn-g1"])
        assert first is None

        second = index.delete(ids=["dn-g1"])
        assert second is None

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# context-manager — REST sync + gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_index_context_manager_rest(client: Pinecone) -> None:
    """REST Index supports the context manager protocol.

    Verifies unified-async-0002 (sync equivalent): the REST index client
    implements __enter__ / __exit__ for automatic resource cleanup.

    - 'with idx as rest_idx:' makes operations available inside the block
    - __enter__ returns the index object itself (not a copy)
    - describe_index_stats() works normally inside the context
    - After the with-block exits (__exit__ calls close()), calling close()
      again must not raise (idempotent resource release)

    Area tag: context-manager
    Transport: rest
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        rest_idx = client.index(name=name)
        with rest_idx as idx:
            # __enter__ must return the same object
            assert idx is rest_idx, "__enter__ must return self"
            assert isinstance(idx, Index)
            # Operations work inside the context
            stats = idx.describe_index_stats()
            assert isinstance(stats, DescribeIndexStatsResponse)
            assert isinstance(stats.total_vector_count, int)
        # After __exit__ called close(), calling close() again must not raise
        rest_idx.close()

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_index_context_manager_grpc(client: Pinecone) -> None:
    """GrpcIndex supports the context manager protocol.

    Verifies unified-async-0002 (gRPC equivalent): GrpcIndex implements
    __enter__ / __exit__ so it can be used as a context manager.

    Area tag: context-manager
    Transport: grpc
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        grpc_idx = client.index(name=name, grpc=True)
        with grpc_idx as gidx:
            assert gidx is grpc_idx, "__enter__ must return self"
            assert isinstance(gidx, GrpcIndex)
            stats = gidx.describe_index_stats()
            assert isinstance(stats, DescribeIndexStatsResponse)
            assert isinstance(stats.total_vector_count, int)
        grpc_idx.close()

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# grpc-async-futures — gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_grpc_async_futures_upsert_query_fetch_grpc(client: Pinecone) -> None:
    """GrpcIndex.*_async() methods return PineconeFuture objects that resolve to
    the correct response types against a live API.

    Verifies unified-grpc-0003: "Can execute gRPC operations asynchronously
    using futures."

    Test sequence:
      1. Submit two parallel upsert_async() futures and collect results via
         as_completed(); each future resolves to a UpsertResponse.
      2. Poll until vectors are queryable (eventual consistency).
      3. Submit a query_async() future and resolve it; verify QueryResponse
         has matches with correct structure.
      4. Submit a fetch_async() future and resolve it; verify FetchResponse
         contains the requested IDs.

    Area tag: grpc-async-futures
    Transport: grpc
    Claim: unified-grpc-0003
    """
    name = unique_name("idx")
    namespace = "fut-ns"
    try:
        client.indexes.create(
            name=name,
            dimension=3,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        grpc_idx = client.index(name=name, grpc=True)

        # --- 1. upsert_async() — two parallel futures ---
        batch_a = [
            {"id": "fut-a1", "values": [0.1, 0.2, 0.3]},
            {"id": "fut-a2", "values": [0.4, 0.5, 0.6]},
        ]
        batch_b = [
            {"id": "fut-b1", "values": [0.7, 0.8, 0.9]},
            {"id": "fut-b2", "values": [0.9, 0.8, 0.7]},
        ]
        fut_a: PineconeFuture[UpsertResponse] = grpc_idx.upsert_async(
            vectors=batch_a, namespace=namespace
        )
        fut_b: PineconeFuture[UpsertResponse] = grpc_idx.upsert_async(
            vectors=batch_b, namespace=namespace
        )

        # Collect results via as_completed; use a generous timeout for live API
        upsert_futures = [fut_a, fut_b]
        for fut in as_completed(upsert_futures, timeout=60):
            result = fut.result(timeout=30.0)
            assert isinstance(result, UpsertResponse), (
                f"Expected UpsertResponse, got {type(result)}"
            )
            assert isinstance(result.upserted_count, int)
            assert result.upserted_count >= 0

        # --- 2. Poll until vectors are queryable (eventual consistency) ---
        poll_until(
            query_fn=lambda: grpc_idx.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 4,
            timeout=120,
            description="all 4 futures-upserted vectors indexed",
        )

        # --- 3. query_async() — submit future, resolve, verify QueryResponse ---
        qfut: PineconeFuture[QueryResponse] = grpc_idx.query_async(
            vector=[0.1, 0.2, 0.3],
            top_k=4,
            namespace=namespace,
        )
        q_result = qfut.result(timeout=30.0)
        assert isinstance(q_result, QueryResponse), (
            f"Expected QueryResponse from query_async(), got {type(q_result)}"
        )
        assert isinstance(q_result.matches, list)
        assert len(q_result.matches) >= 1, "query_async() must return at least one match"
        # Verify match structure; scores may vary due to ANN approximation
        upserted_ids = {"fut-a1", "fut-a2", "fut-b1", "fut-b2"}
        for m in q_result.matches:
            assert isinstance(m.id, str)
            assert isinstance(m.score, float)
            assert m.id in upserted_ids, f"Unexpected match ID {m.id!r}"

        # --- 4. fetch_async() — submit future, resolve, verify FetchResponse ---
        ffut: PineconeFuture[FetchResponse] = grpc_idx.fetch_async(
            ids=["fut-a1", "fut-b1"],
            namespace=namespace,
        )
        f_result = ffut.result(timeout=30.0)
        assert isinstance(f_result, FetchResponse), (
            f"Expected FetchResponse from fetch_async(), got {type(f_result)}"
        )
        assert isinstance(f_result.vectors, dict)
        assert "fut-a1" in f_result.vectors, "fut-a1 must be present in fetch_async() result"
        assert "fut-b1" in f_result.vectors, "fut-b1 must be present in fetch_async() result"

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# fetch-nonexistent — REST sync + gRPC
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_nonexistent_ids_returns_empty_vectors_rest(client: Pinecone) -> None:
    """Fetching IDs that were never upserted returns an empty vectors map, not an error.

    Verifies unified-vec-0053: "Fetching IDs that do not exist returns an empty
    vectors map rather than an error."

    Area tag: fetch-nonexistent
    Transport: rest
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=3,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        idx = client.index(name=name)

        # Upsert one real vector to establish the namespace
        idx.upsert(vectors=[{"id": "real-v1", "values": [0.1, 0.2, 0.3]}])
        poll_until(
            query_fn=lambda: idx.fetch(ids=["real-v1"]),
            check_fn=lambda r: "real-v1" in r.vectors,
            timeout=120,
            description="real-v1 fetchable after upsert",
        )

        # Fetch IDs that were never upserted — should return empty dict, not raise
        result = idx.fetch(ids=["never-upserted-aaa", "never-upserted-bbb"])

        assert isinstance(result, FetchResponse)
        assert isinstance(result.vectors, dict)
        # Non-existent IDs are simply absent — no error raised
        assert "never-upserted-aaa" not in result.vectors
        assert "never-upserted-bbb" not in result.vectors

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_fetch_nonexistent_ids_returns_empty_vectors_grpc(client: Pinecone) -> None:
    """Fetching IDs that were never upserted returns an empty vectors map via gRPC.

    Verifies unified-vec-0053 on the gRPC transport.

    Area tag: fetch-nonexistent
    Transport: grpc
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=3,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        idx = client.index(name=name, grpc=True)

        # Upsert one real vector to establish the namespace
        idx.upsert(vectors=[{"id": "real-v1", "values": [0.1, 0.2, 0.3]}])
        poll_until(
            query_fn=lambda: idx.fetch(ids=["real-v1"]),
            check_fn=lambda r: "real-v1" in r.vectors,
            timeout=120,
            description="real-v1 fetchable after upsert (gRPC)",
        )

        # Fetch IDs that were never upserted — should return empty dict, not raise
        result = idx.fetch(ids=["never-upserted-aaa", "never-upserted-bbb"])

        assert isinstance(result, FetchResponse)
        assert isinstance(result.vectors, dict)
        assert "never-upserted-aaa" not in result.vectors
        assert "never-upserted-bbb" not in result.vectors

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# grpc-delete-async — gRPC only
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_grpc_delete_async_future_resolves_to_none(client: Pinecone) -> None:
    """GrpcIndex.delete_async() returns a PineconeFuture[None] that resolves to None.

    Verifies unified-grpc-0003: "Can execute gRPC operations asynchronously
    using futures." — specifically the delete_async() variant not covered by
    ET-078 (which covered upsert_async, query_async, fetch_async).

    Test sequence:
    1. Upsert 3 vectors via sync gRPC upsert().
    2. Poll until all 3 vectors are fetchable.
    3. delete_async(ids=[...]) for 2 of the 3 → future.result() must be None.
    4. Poll until the 2 deleted vectors are absent from fetch().
    5. Verify the 3rd vector is still present (partial delete correctness).
    6. delete_async(delete_all=True) → future.result() must be None.
    7. Poll until namespace is empty.

    Area tag: grpc-delete-async
    Transport: grpc
    Claim: unified-grpc-0003
    """
    name = unique_name("idx")
    namespace = "da-ns"
    try:
        client.indexes.create(
            name=name,
            dimension=3,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        grpc_idx = client.index(name=name, grpc=True)

        # --- 1. Upsert 3 vectors via sync gRPC upsert() ---
        vectors = [
            {"id": "da-v1", "values": [0.1, 0.2, 0.3]},
            {"id": "da-v2", "values": [0.4, 0.5, 0.6]},
            {"id": "da-v3", "values": [0.7, 0.8, 0.9]},
        ]
        grpc_idx.upsert(vectors=vectors, namespace=namespace)

        # --- 2. Poll until all 3 vectors are fetchable ---
        poll_until(
            query_fn=lambda: grpc_idx.fetch(ids=["da-v1", "da-v2", "da-v3"], namespace=namespace),
            check_fn=lambda r: len(r.vectors) == 3,
            timeout=120,
            description="all 3 vectors fetchable after upsert (gRPC delete_async test)",
        )

        # --- 3. delete_async(ids=[...]) for 2 of the 3 vectors ---
        del_future: PineconeFuture[None] = grpc_idx.delete_async(
            ids=["da-v1", "da-v2"],
            namespace=namespace,
        )
        # future.result() must resolve to None (delete returns no content)
        del_result = del_future.result(timeout=30.0)
        assert del_result is None, (
            f"delete_async().result() must be None, got {del_result!r}"
        )

        # --- 4. Poll until the 2 deleted vectors are absent ---
        poll_until(
            query_fn=lambda: grpc_idx.fetch(ids=["da-v1", "da-v2"], namespace=namespace),
            check_fn=lambda r: len(r.vectors) == 0,
            timeout=120,
            description="da-v1 and da-v2 absent after delete_async()",
        )

        # --- 5. Verify da-v3 is still present (partial delete) ---
        remaining = grpc_idx.fetch(ids=["da-v3"], namespace=namespace)
        assert isinstance(remaining, FetchResponse)
        assert "da-v3" in remaining.vectors, "da-v3 should survive partial delete_async()"

        # --- 6. delete_async(delete_all=True) clears the namespace ---
        all_del_future: PineconeFuture[None] = grpc_idx.delete_async(
            delete_all=True,
            namespace=namespace,
        )
        all_del_result = all_del_future.result(timeout=30.0)
        assert all_del_result is None, (
            f"delete_async(delete_all=True).result() must be None, got {all_del_result!r}"
        )

        # --- 7. Poll until namespace is empty ---
        poll_until(
            query_fn=lambda: grpc_idx.fetch(ids=["da-v3"], namespace=namespace),
            check_fn=lambda r: len(r.vectors) == 0,
            timeout=120,
            description="namespace empty after delete_async(delete_all=True)",
        )

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# namespace creation error paths — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_namespace_error_paths_rest(client: Pinecone) -> None:
    """create_namespace() rejects invalid names client-side and raises ConflictError for duplicates.

    Verifies claims:
    - unified-ns-0010: Namespace creation is rejected when name is empty or whitespace-only.
    - unified-ns-0012: Creating a namespace that already exists raises a ConflictError (HTTP 409).

    No integration test covered these error paths; only unit tests (test-ns-0009) exercised them.
    """
    name = unique_name("idx")
    ns_name = "cnep-ns-alpha"
    index = None
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        # unified-ns-0010: empty name → client-side PineconeValueError (no API call made)
        with pytest.raises(PineconeValueError):
            index.create_namespace(name="")

        # unified-ns-0010: whitespace-only name → client-side PineconeValueError
        with pytest.raises(PineconeValueError):
            index.create_namespace(name="   ")

        # Precondition for ns-0012: create the namespace successfully
        created = index.create_namespace(name=ns_name)
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name

        # unified-ns-0012: creating the same namespace again raises ConflictError (409)
        with pytest.raises(ConflictError) as exc_info:
            index.create_namespace(name=ns_name)
        assert exc_info.value.status_code == 409

    finally:
        if index is not None:
            cleanup_resource(lambda: index.delete_namespace(name=ns_name), ns_name, "namespace")
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# namespace creation with schema — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_namespace_with_schema_rest(client: Pinecone) -> None:
    """create_namespace() accepts an optional schema dict and creates the namespace successfully.

    Verifies:
    - unified-ns-0001: Can create a named namespace, optionally providing a schema configuration.

    The existing test_namespace_crud_lifecycle_rest only calls create_namespace(name=...) without
    a schema parameter. This test exercises the schema= code path in the SDK: when schema is not
    None, the SDK adds body["schema"] = schema to the POST /namespaces request. No integration test
    previously exercised this path.
    """
    name = unique_name("idx")
    ns_name = "schema-ns-alpha"
    index = None
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        # Create a namespace with a schema specifying that "genre" is filterable
        created = index.create_namespace(
            name=ns_name,
            schema={"fields": {"genre": {"filterable": True}}},
        )

        # Verify response type and structure
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name
        assert created.record_count == 0  # new namespace has no vectors

        # describe_namespace returns the namespace as accessible (schema was accepted)
        described = index.describe_namespace(name=ns_name)
        assert isinstance(described, NamespaceDescription)
        assert described.name == ns_name
        assert isinstance(described.record_count, int)
        assert described.record_count == 0

    finally:
        if index is not None:
            cleanup_resource(lambda: index.delete_namespace(name=ns_name), ns_name, "namespace")
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# describe_namespace record_count after upsert — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_namespace_record_count_updates_after_upsert_rest(client: Pinecone) -> None:
    """describe_namespace().record_count reflects the vector count after upsert (REST sync).

    Verifies claim unified-ns-0003: "Can describe a namespace by name, returning its
    record count and schema." The record_count must accurately track the number of
    vectors in the namespace — not just be 0 at creation time.

    Operation sequence tested:
    1. Create namespace → verify record_count == 0
    2. Upsert 4 vectors into the namespace
    3. Poll describe_namespace() until record_count > 0 (eventual consistency)
    4. Verify record_count == 4

    No existing test verifies the state transition from 0 to N for record_count.
    """
    name = unique_name("idx")
    ns_name = "rcnt-ns-alpha"
    index = None
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        # 1. Create namespace explicitly — record_count starts at 0
        created = index.create_namespace(name=ns_name)
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name
        assert created.record_count == 0, (
            f"Freshly created namespace should have record_count == 0, got {created.record_count}"
        )

        # 2. Upsert 4 vectors into the namespace
        upsert_resp = index.upsert(
            vectors=[
                {"id": "rcnt-v1", "values": [0.1, 0.2]},
                {"id": "rcnt-v2", "values": [0.3, 0.4]},
                {"id": "rcnt-v3", "values": [0.5, 0.6]},
                {"id": "rcnt-v4", "values": [0.7, 0.8]},
            ],
            namespace=ns_name,
        )
        assert isinstance(upsert_resp, UpsertResponse)
        assert upsert_resp.upserted_count == 4

        # 3. Poll until describe_namespace reports at least 4 vectors (eventual consistency)
        final = poll_until(
            query_fn=lambda: index.describe_namespace(name=ns_name),
            check_fn=lambda r: r.record_count >= 4,
            timeout=120,
            description="describe_namespace record_count reaches 4 after upsert",
        )
        assert isinstance(final, NamespaceDescription)

        # 4. Verify the record_count equals the number of vectors upserted
        assert final.record_count == 4, (
            f"Expected record_count == 4 after upserting 4 vectors, got {final.record_count}"
        )
        assert final.name == ns_name

    finally:
        if index is not None:
            cleanup_resource(lambda: index.delete_namespace(name=ns_name), ns_name, "namespace")
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")
