"""Integration tests for data-plane vector operations (sync REST + gRPC)."""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.responses import DescribeIndexStatsResponse, FetchResponse, ListItem, ListResponse, NamespaceSummary, QueryResponse, UpdateResponse, UpsertResponse
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
# update-vectors — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0002: pinecone._grpc Rust extension not installed; ModuleNotFoundError on GrpcIndex creation",
)
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
                "upd-v1" in r.vectors
                and abs(r.vectors["upd-v1"].values[0] - 0.9) < 1e-4
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
# describe-stats — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0002: pinecone._grpc Rust extension not installed; ModuleNotFoundError on GrpcIndex creation",
)
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
    NAMED_NS = "ns-alpha"
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
            namespace=NAMED_NS,
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
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace=NAMED_NS),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="named namespace vectors queryable",
        )

        # Query in the named namespace
        ns_query = index.query(vector=[0.1, 0.2], top_k=10, namespace=NAMED_NS)
        assert isinstance(ns_query, QueryResponse)
        assert ns_query.namespace == NAMED_NS
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
        assert NAMED_NS in stats.namespaces
        assert stats.namespaces[NAMED_NS].vector_count == 2
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# namespaces — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0002: pinecone._grpc Rust extension not installed; ModuleNotFoundError on GrpcIndex creation",
)
def test_namespaces_grpc(client: Pinecone) -> None:
    """Upsert to named namespace via GrpcIndex and query within it."""
    name = unique_name("idx")
    NAMED_NS = "ns-alpha"
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
            namespace=NAMED_NS,
        )

        result = index.query(vector=[0.1, 0.2], top_k=10, namespace=NAMED_NS)
        assert isinstance(result, QueryResponse)
        assert result.namespace == NAMED_NS
        ids = {m.id for m in result.matches}
        assert "ns-v1" in ids or "ns-v2" in ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")
