"""Integration tests for data-plane vector operations (async REST)."""

from __future__ import annotations

import pytest

from pinecone import AsyncIndex, AsyncPinecone
from pinecone.models.indexes.specs import ServerlessSpec
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
from tests.integration.conftest import async_cleanup_resource, async_poll_until, unique_name

# ---------------------------------------------------------------------------
# delete-vectors — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_vectors_rest_async(async_client: AsyncPinecone) -> None:
    """Delete vectors by IDs via AsyncIndex (REST) and verify they are gone."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        await idx.upsert(
            vectors=[
                {"id": "del-v1", "values": [0.1, 0.2]},
                {"id": "del-v2", "values": [0.3, 0.4]},
                {"id": "del-v3", "values": [0.5, 0.6]},
            ]
        )

        # Wait for all 3 vectors to be fetchable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.fetch(ids=["del-v1", "del-v2", "del-v3"]),
            check_fn=lambda r: len(r.vectors) == 3,
            timeout=120,
            description="all 3 vectors fetchable before delete",
        )

        # Delete just v1 and v2 by IDs
        result = await idx.delete(ids=["del-v1", "del-v2"])
        assert result is None  # delete returns None on success

        # Wait until deleted vectors are gone (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.fetch(ids=["del-v1", "del-v2"]),
            check_fn=lambda r: len(r.vectors) == 0,
            timeout=120,
            description="deleted vectors gone after delete",
        )

        # Verify v3 is still present
        remaining = await idx.fetch(ids=["del-v3"])
        assert isinstance(remaining, FetchResponse)
        assert "del-v3" in remaining.vectors
        assert "del-v1" not in remaining.vectors
        assert "del-v2" not in remaining.vectors
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )

# ---------------------------------------------------------------------------
# upsert — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_vectors_rest_async(async_client: AsyncPinecone) -> None:
    """Upsert vectors via AsyncIndex (REST) and verify upserted_count."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # describe() caches the host so pc.index(name=...) can resolve it
        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        result = await idx.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
                {"id": "v3", "values": [0.5, 0.6]},
            ]
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 3
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# query — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_by_vector_rest_async(async_client: AsyncPinecone) -> None:
    """Query by vector via AsyncIndex (REST) and verify matches structure."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        await idx.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
                {"id": "v3", "values": [0.5, 0.6]},
            ]
        )

        # Wait for all 3 vectors to be queryable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=3),
            check_fn=lambda r: len(r.matches) == 3,
            timeout=120,
            description="all 3 vectors queryable after upsert",
        )

        result = await idx.query(vector=[0.1, 0.2], top_k=2)

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
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# fetch — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_vectors_rest_async(async_client: AsyncPinecone) -> None:
    """Fetch vectors by ID via AsyncIndex (REST) and verify returned vector data."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        await idx.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
                {"id": "v3", "values": [0.5, 0.6]},
            ]
        )

        # Wait for vectors to be fetchable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.fetch(ids=["v1", "v2", "v3"]),
            check_fn=lambda r: len(r.vectors) == 3,
            timeout=120,
            description="all 3 vectors fetchable after upsert",
        )

        result = await idx.fetch(ids=["v1", "v2"])

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
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# list-vectors — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_vectors_rest_async(async_client: AsyncPinecone) -> None:
    """List vectors via AsyncIndex (REST) and verify pagination structure and IDs."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        await idx.upsert(
            vectors=[
                {"id": "lst-v1", "values": [0.1, 0.2]},
                {"id": "lst-v2", "values": [0.3, 0.4]},
                {"id": "lst-v3", "values": [0.5, 0.6]},
            ]
        )

        # Wait for all 3 vectors to appear in list results (eventual consistency)
        async def _collect_ids() -> list[str]:
            return [
                item.id
                async for page in idx.list(prefix="lst-")  # type: ignore[union-attr]
                for item in page.vectors
                if item.id is not None
            ]

        await async_poll_until(
            query_fn=_collect_ids,
            check_fn=lambda ids: len(ids) >= 3,
            timeout=120,
            description="all 3 vectors listable after upsert",
        )

        # Collect all pages and verify structure
        pages: list[ListResponse] = [page async for page in idx.list(prefix="lst-")]

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
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# update-vectors — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_vectors_rest_async(async_client: AsyncPinecone) -> None:
    """Update a vector's values via AsyncIndex (REST) and verify the change is reflected."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        await idx.upsert(
            vectors=[
                {"id": "upd-v1", "values": [0.1, 0.2]},
                {"id": "upd-v2", "values": [0.3, 0.4]},
            ]
        )

        # Wait for vectors to be fetchable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.fetch(ids=["upd-v1", "upd-v2"]),  # type: ignore[union-attr]
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="vectors fetchable before update",
        )

        # Update upd-v1 with new values
        result = await idx.update(id="upd-v1", values=[0.9, 0.8])

        assert isinstance(result, UpdateResponse)
        # The update API returns {} on success; matched_records may be None
        assert result.matched_records is None or isinstance(result.matched_records, int)

        # Poll until the updated values are reflected
        await async_poll_until(
            query_fn=lambda: idx.fetch(ids=["upd-v1"]),  # type: ignore[union-attr]
            check_fn=lambda r: (
                "upd-v1" in r.vectors
                and len(r.vectors["upd-v1"].values) == 2
                and abs(r.vectors["upd-v1"].values[0] - 0.9) < 1e-4
            ),
            timeout=120,
            description="updated values reflected in fetch",
        )

        # Verify upd-v2 was not modified
        check = await idx.fetch(ids=["upd-v2"])
        assert abs(check.vectors["upd-v2"].values[0] - 0.3) < 1e-4
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# describe-stats — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_describe_index_stats_rest_async(async_client: AsyncPinecone) -> None:
    """Call describe_index_stats() via AsyncIndex (REST) and verify response structure."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=3,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        # Upsert a few vectors so stats are non-trivial
        await idx.upsert(
            vectors=[
                {"id": "st-v1", "values": [0.1, 0.2, 0.3]},
                {"id": "st-v2", "values": [0.4, 0.5, 0.6]},
            ]
        )

        # Wait until at least 1 vector is counted in stats (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.describe_index_stats(),  # type: ignore[union-attr]
            check_fn=lambda r: r.total_vector_count >= 1,
            timeout=120,
            description="at least 1 vector counted in stats after upsert",
        )

        stats = await idx.describe_index_stats()

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
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# namespaces — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_namespaces_rest_async(async_client: AsyncPinecone) -> None:
    """Upsert to named namespace via AsyncIndex (REST) and query within it."""
    name = unique_name("idx")
    named_ns = "ns-alpha"
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        idx = async_client.index(name=name)

        # Upsert vectors into named namespace
        ns_result = await idx.upsert(
            vectors=[
                {"id": "ns-v1", "values": [0.1, 0.2]},
                {"id": "ns-v2", "values": [0.3, 0.4]},
            ],
            namespace=named_ns,
        )
        assert isinstance(ns_result, UpsertResponse)
        assert ns_result.upserted_count == 2

        # Upsert different vectors into the default namespace
        def_result = await idx.upsert(
            vectors=[
                {"id": "def-v1", "values": [0.9, 0.8]},
            ],
            namespace="",
        )
        assert isinstance(def_result, UpsertResponse)
        assert def_result.upserted_count == 1

        # Wait until ns-alpha vectors are queryable in the named namespace
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns),  # type: ignore[union-attr]
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="named namespace vectors queryable",
        )

        # Query in the named namespace
        ns_query = await idx.query(vector=[0.1, 0.2], top_k=10, namespace=named_ns)
        assert isinstance(ns_query, QueryResponse)
        assert ns_query.namespace == named_ns
        ns_ids = {m.id for m in ns_query.matches}
        assert "ns-v1" in ns_ids
        assert "ns-v2" in ns_ids
        # Default namespace vectors must NOT appear in named-namespace query
        assert "def-v1" not in ns_ids

        # Verify stats shows named namespace
        await async_poll_until(
            query_fn=lambda: idx.describe_index_stats(),  # type: ignore[union-attr]
            check_fn=lambda s: s.total_vector_count >= 3,
            timeout=120,
            description="stats reflect all 3 vectors",
        )
        stats = await idx.describe_index_stats()
        assert isinstance(stats.namespaces, dict)
        assert named_ns in stats.namespaces
        assert stats.namespaces[named_ns].vector_count == 2
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# list_paginated — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_paginated_returns_single_page_rest_async(
    async_client: AsyncPinecone,
) -> None:
    """list_paginated() via AsyncIndex returns one page; limit is respected; no token on last page."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        # Upsert 4 vectors with a shared prefix
        await idx.upsert(
            vectors=[
                {"id": "pg-v1", "values": [0.1, 0.2]},
                {"id": "pg-v2", "values": [0.3, 0.4]},
                {"id": "pg-v3", "values": [0.5, 0.6]},
                {"id": "pg-v4", "values": [0.7, 0.8]},
            ]
        )

        # Wait until all 4 vectors appear in list results
        await async_poll_until(
            query_fn=lambda: idx.list_paginated(prefix="pg-", limit=100),  # type: ignore[union-attr]
            check_fn=lambda r: len(r.vectors) >= 4,
            timeout=120,
            description="all 4 vectors listable after upsert",
        )

        # 1. list_paginated() returns a ListResponse (not an async generator)
        page = await idx.list_paginated(prefix="pg-", limit=100)
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

        # 5. No pagination token on the final page (unified-vec-0056)
        assert page.pagination is None or page.pagination.next is None

        # 6. limit=2 returns at most 2 items
        limited_page = await idx.list_paginated(prefix="pg-", limit=2)
        assert isinstance(limited_page, ListResponse)
        assert len(limited_page.vectors) <= 2
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )
