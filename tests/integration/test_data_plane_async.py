"""Integration tests for data-plane vector operations (async REST)."""

from __future__ import annotations

import pytest

from pinecone import AsyncIndex, AsyncPinecone
from pinecone.errors import ApiError, ConflictError, PineconeValueError
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
from tests.integration.conftest import async_cleanup_resource, async_poll_until, unique_name

# ---------------------------------------------------------------------------
# delete-vectors — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
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
@pytest.mark.anyio
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
@pytest.mark.anyio
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
@pytest.mark.anyio
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
@pytest.mark.anyio
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
@pytest.mark.anyio
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
@pytest.mark.anyio
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
@pytest.mark.anyio
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
@pytest.mark.anyio
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


# ---------------------------------------------------------------------------
# describe-stats with filter — serverless rejects (REST async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_describe_index_stats_filter_unsupported_on_serverless_rest_async(
    async_client: AsyncPinecone,
) -> None:
    """Verify describe_index_stats(filter=...) raises ApiError(400) on a serverless index (async)."""
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

        # Upsert one vector so the index has some content
        await idx.upsert(vectors=[{"id": "fa-v1", "values": [0.1, 0.2, 0.3]}])

        # The filter parameter is not supported on serverless/starter indexes —
        # the API returns 400 and the SDK should surface it as ApiError.
        with pytest.raises(ApiError) as exc_info:
            await idx.describe_index_stats(filter={"tag": {"$eq": "a"}})

        assert exc_info.value.status_code == 400
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# namespace CRUD — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_namespace_crud_lifecycle_rest_async(async_client: AsyncPinecone) -> None:
    """Async create_namespace / describe_namespace / list_namespaces_paginated / delete_namespace.

    Verifies claims:
    - unified-ns-0001: Can create a named namespace.
    - unified-ns-0002: Creation returns name and record_count == 0.
    - unified-ns-0003: Can describe a namespace by name.
    - unified-ns-0004: Can delete a namespace by name.
    - unified-ns-0005: Can list all namespaces with optional prefix filtering.
    - unified-ns-0008: Namespace list response omits pagination token on the final page.
    """
    name = unique_name("idx")
    ns_name = "crud-ns-beta"
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

        # 1. Create namespace — returns NamespaceDescription with record_count == 0
        created = await idx.create_namespace(name=ns_name)
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name
        assert created.record_count == 0  # unified-ns-0002

        # 2. Describe namespace — returns NamespaceDescription
        described = await idx.describe_namespace(name=ns_name)
        assert isinstance(described, NamespaceDescription)
        assert described.name == ns_name
        assert isinstance(described.record_count, int)

        # 3. Namespace appears in list_namespaces_paginated with prefix match
        list_resp = await idx.list_namespaces_paginated(prefix="crud-ns-", limit=100)
        assert isinstance(list_resp, ListNamespacesResponse)
        ns_names = [ns.name for ns in list_resp.namespaces]
        assert ns_name in ns_names

        # Each entry is a NamespaceDescription with string name and int record_count
        for ns in list_resp.namespaces:
            assert isinstance(ns, NamespaceDescription)
            assert isinstance(ns.name, str)
            assert isinstance(ns.record_count, int)

        # 4. Pagination token absent on the final page (unified-ns-0008)
        assert list_resp.pagination is None or list_resp.pagination.next is None

        # 5. Delete namespace — returns None on success
        result = await idx.delete_namespace(name=ns_name)
        assert result is None  # unified-ns-0004

        # 6. After deletion, namespace no longer appears in listing
        post_delete = await idx.list_namespaces_paginated(prefix="crud-ns-", limit=100)
        assert isinstance(post_delete, ListNamespacesResponse)
        post_names = [ns.name for ns in post_delete.namespaces]
        assert ns_name not in post_names
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# list_namespaces generator — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_namespaces_generator_rest_async(async_client: AsyncPinecone) -> None:
    """list_namespaces() async generator yields ListNamespacesResponse pages with
    NamespaceDescription items; generator follows pagination tokens automatically.

    Verifies claims:
    - unified-ns-0007: The namespace list generator yields NamespaceDescription objects per page.
    - unified-ns-0005: Can list all namespaces with optional prefix filtering and pagination.
    """
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    ns_a = "lnsgen-ns-a"
    ns_b = "lnsgen-ns-b"
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        idx = async_client.index(name=name)

        # Upsert one vector into each namespace to create them implicitly
        await idx.upsert(vectors=[{"id": "lnsg-v1", "values": [0.1, 0.2]}], namespace=ns_a)
        await idx.upsert(vectors=[{"id": "lnsg-v2", "values": [0.3, 0.4]}], namespace=ns_b)

        # Poll until both namespaces appear in list_namespaces_paginated
        await async_poll_until(
            query_fn=lambda: idx.list_namespaces_paginated(prefix="lnsgen-ns-", limit=100),
            check_fn=lambda r: len(r.namespaces) >= 2,
            timeout=120,
            description="both lnsgen-ns-* namespaces appear via list_namespaces_paginated",
        )

        # --- Exercise the async generator ---
        pages: list[ListNamespacesResponse] = []
        async for page in idx.list_namespaces(prefix="lnsgen-ns-"):
            pages.append(page)

        assert len(pages) >= 1, "list_namespaces() async generator must yield at least one page"

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
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# list_paginated multi-page — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_paginated_multi_page_rest_async(async_client: AsyncPinecone) -> None:
    """list_paginated() with limit=2 returns a token when more pages exist; following the token
    reaches the next page; the final page has no token (async variant).

    Verifies claims:
    - unified-vec-0030: paginated list method returns a single page (caller must follow token)
    - unified-vec-0056: list-paginated returns no pagination token on the final page
    - unified-pag-0002: vector listing supports cursor-based pagination via single-page method
    """
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
                {"id": "mp-v1", "values": [0.1, 0.2]},
                {"id": "mp-v2", "values": [0.3, 0.4]},
                {"id": "mp-v3", "values": [0.5, 0.6]},
                {"id": "mp-v4", "values": [0.7, 0.8]},
            ]
        )

        # Wait until all 4 vectors appear in list results
        async def _list_all() -> ListResponse:
            return await idx.list_paginated(prefix="mp-", limit=100)  # type: ignore[union-attr]

        await async_poll_until(
            query_fn=_list_all,
            check_fn=lambda r: len(r.vectors) >= 4,
            timeout=120,
            description="all 4 vectors listable after upsert",
        )

        # Traverse all pages manually using limit=2 (forces at least 2 pages)
        all_ids: list[str] = []
        token: str | None = None
        pages_seen = 0

        while True:
            page = await idx.list_paginated(prefix="mp-", limit=2, pagination_token=token)
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
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# delete-nonexistent-ids — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_delete_nonexistent_ids_returns_none_async(async_client: AsyncPinecone) -> None:
    """Delete with IDs that were never upserted (or already deleted) returns None.

    Verifies unified-vec-0032: "Deleting vectors does not raise an error when
    the specified IDs do not exist."
    """
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
        idx = async_client.index(name=name)

        # Establish namespace by upserting a sentinel vector first
        await idx.upsert(vectors=[{"id": "dn-a1", "values": [0.3, 0.7]}])
        await async_poll_until(
            query_fn=lambda: idx.fetch(ids=["dn-a1"]),
            check_fn=lambda r: len(r.vectors) == 1,
            timeout=120,
            description="dn-a1 fetchable (namespace established for async test)",
        )

        # Sub-case 1: delete IDs that were never upserted (namespace now exists)
        result = await idx.delete(ids=["never-existed-async-x", "never-existed-async-y"])
        assert result is None

        # Sub-case 2: delete dn-a1 (exists), then delete again (already gone — idempotency)
        first = await idx.delete(ids=["dn-a1"])
        assert first is None

        second = await idx.delete(ids=["dn-a1"])
        assert second is None

    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# context-manager — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_index_context_manager_async(async_client: AsyncPinecone) -> None:
    """AsyncIndex supports the async context manager protocol.

    Verifies unified-async-0002: the async index client implements __aenter__
    and __aexit__ for automatic resource cleanup.

    - 'async with idx as ai:' allows operations inside the block
    - __aenter__ returns the index object itself (not a copy)
    - describe_index_stats() works normally inside the context
    - After the with-block exits (__aexit__ calls close()), calling close()
      again must not raise (idempotent resource release)

    Area tag: context-manager
    Transport: rest-async
    """
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        async_index = async_client.index(name=name)
        async with async_index as ai:
            # __aenter__ must return the same object
            assert ai is async_index, "__aenter__ must return self"
            assert isinstance(ai, AsyncIndex)
            # Operations work inside the context
            stats = await ai.describe_index_stats()
            assert isinstance(stats, DescribeIndexStatsResponse)
            assert isinstance(stats.total_vector_count, int)
        # After __aexit__ called close(), calling close() again must not raise
        await async_index.close()

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# fetch-nonexistent — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_fetch_nonexistent_ids_returns_empty_vectors_async(
    async_client: AsyncPinecone,
) -> None:
    """Fetching IDs that were never upserted returns an empty vectors map, not an error.

    Verifies unified-vec-0053: "Fetching IDs that do not exist returns an empty
    vectors map rather than an error."

    Area tag: fetch-nonexistent
    Transport: rest-async
    """
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
        idx = async_client.index(name=name)

        # Upsert one real vector to establish the namespace
        await idx.upsert(vectors=[{"id": "real-v1", "values": [0.1, 0.2, 0.3]}])
        await async_poll_until(
            query_fn=lambda: idx.fetch(ids=["real-v1"]),
            check_fn=lambda r: "real-v1" in r.vectors,
            timeout=120,
            description="real-v1 fetchable after upsert (async)",
        )

        # Fetch IDs that were never upserted — should return empty dict, not raise
        result = await idx.fetch(ids=["never-upserted-aaa", "never-upserted-bbb"])

        assert isinstance(result, FetchResponse)
        assert isinstance(result.vectors, dict)
        # Non-existent IDs are simply absent — no error raised
        assert "never-upserted-aaa" not in result.vectors
        assert "never-upserted-bbb" not in result.vectors

    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# namespace creation error paths — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_namespace_error_paths_async(async_client: AsyncPinecone) -> None:
    """create_namespace() rejects invalid names client-side and raises ConflictError for duplicates.

    Verifies claims:
    - unified-ns-0010: Namespace creation is rejected when name is empty or whitespace-only.
    - unified-ns-0012: Creating a namespace that already exists raises a ConflictError (HTTP 409).

    Async transport parity for test_create_namespace_error_paths_rest.
    """
    name = unique_name("idx")
    ns_name = "cnep-ns-beta"
    idx = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        idx = async_client.index(name=name)

        # unified-ns-0010: empty name → client-side PineconeValueError (no API call made)
        with pytest.raises(PineconeValueError):
            await idx.create_namespace(name="")

        # unified-ns-0010: whitespace-only name → client-side PineconeValueError
        with pytest.raises(PineconeValueError):
            await idx.create_namespace(name="   ")

        # Precondition for ns-0012: create the namespace successfully
        created = await idx.create_namespace(name=ns_name)
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name

        # unified-ns-0012: creating the same namespace again raises ConflictError (409)
        with pytest.raises(ConflictError) as exc_info:
            await idx.create_namespace(name=ns_name)
        assert exc_info.value.status_code == 409

    finally:
        if idx is not None:
            await async_cleanup_resource(
                lambda: idx.delete_namespace(name=ns_name), ns_name, "namespace"
            )
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# namespace creation with schema — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_namespace_with_schema_async(async_client: AsyncPinecone) -> None:
    """create_namespace() accepts an optional schema dict and creates the namespace successfully (async).

    Verifies:
    - unified-ns-0001: Can create a named namespace, optionally providing a schema configuration.

    Async transport parity for test_create_namespace_with_schema_rest.
    The schema= path in the SDK sends body["schema"] = schema in the POST /namespaces request.
    """
    name = unique_name("idx")
    ns_name = "schema-ns-beta"
    idx = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        idx = async_client.index(name=name)

        # Create a namespace with a schema specifying that "genre" is filterable
        created = await idx.create_namespace(
            name=ns_name,
            schema={"fields": {"genre": {"filterable": True}}},
        )

        # Verify response type and structure
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name
        assert created.record_count == 0  # new namespace has no vectors

        # describe_namespace returns the namespace as accessible (schema was accepted)
        described = await idx.describe_namespace(name=ns_name)
        assert isinstance(described, NamespaceDescription)
        assert described.name == ns_name
        assert isinstance(described.record_count, int)
        assert described.record_count == 0

    finally:
        if idx is not None:
            await async_cleanup_resource(
                lambda: idx.delete_namespace(name=ns_name), ns_name, "namespace"
            )
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# describe_namespace record_count after upsert — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_describe_namespace_record_count_updates_after_upsert_async(
    async_client: AsyncPinecone,
) -> None:
    """describe_namespace().record_count reflects the vector count after upsert (REST async).

    Verifies claim unified-ns-0003: "Can describe a namespace by name, returning its
    record count and schema." The record_count must accurately track the number of
    vectors in the namespace — not just be 0 at creation time.

    Operation sequence tested:
    1. Create namespace → verify record_count == 0
    2. Upsert 4 vectors into the namespace
    3. Poll describe_namespace() until record_count > 0 (eventual consistency)
    4. Verify record_count == 4

    Async transport parity for test_describe_namespace_record_count_updates_after_upsert_rest.
    """
    name = unique_name("idx")
    ns_name = "rcnt-ns-beta"
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

        # 1. Create namespace explicitly — record_count starts at 0
        created = await idx.create_namespace(name=ns_name)
        assert isinstance(created, NamespaceDescription)
        assert created.name == ns_name
        assert created.record_count == 0, (
            f"Freshly created namespace should have record_count == 0, got {created.record_count}"
        )

        # 2. Upsert 4 vectors into the namespace
        upsert_resp = await idx.upsert(
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
        final = await async_poll_until(
            query_fn=lambda: idx.describe_namespace(name=ns_name),
            check_fn=lambda r: r.record_count >= 4,
            timeout=120,
            description="describe_namespace record_count reaches 4 after upsert (async)",
        )
        assert isinstance(final, NamespaceDescription)

        # 4. Verify the record_count equals the number of vectors upserted
        assert final.record_count == 4, (
            f"Expected record_count == 4 after upserting 4 vectors, got {final.record_count}"
        )
        assert final.name == ns_name

    finally:
        if idx is not None:
            await async_cleanup_resource(
                lambda: idx.delete_namespace(name=ns_name), ns_name, "namespace"
            )
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# list_namespaces multi-page pagination — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_namespaces_multi_page_pagination_async(async_client: AsyncPinecone) -> None:
    """list_namespaces_paginated() with limit=1 forces multi-page results; intermediate
    pages carry a non-None pagination token; the final page has no token (async variant).

    Verifies claims:
    - unified-ns-0008: Namespace list response omits the pagination token on the final page.
    - unified-ns-0005: Can list all namespaces with optional prefix filtering and pagination.
    """
    name = unique_name("idx")
    prefix = "mpns-"
    ns_names = ["mpns-a", "mpns-b", "mpns-c"]
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

        # Upsert one vector per namespace to create them implicitly
        for i, ns in enumerate(ns_names):
            await idx.upsert(
                vectors=[{"id": f"mpns-v{i}", "values": [0.1 * (i + 1), 0.2 * (i + 1)]}],
                namespace=ns,
            )

        # Wait until all 3 namespaces appear (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.list_namespaces_paginated(prefix=prefix, limit=100),
            check_fn=lambda r: len(r.namespaces) >= 3,
            timeout=120,
            description="all 3 mpns-* namespaces visible via list_namespaces_paginated (async)",
        )

        # Traverse pages manually with limit=1 (forces >=3 pages for 3 namespaces)
        collected_names: list[str] = []
        token: str | None = None
        pages_seen = 0

        while True:
            page = await idx.list_namespaces_paginated(
                prefix=prefix, limit=1, pagination_token=token
            )
            assert isinstance(page, ListNamespacesResponse)
            assert len(page.namespaces) <= 1  # limit is respected per page

            for ns in page.namespaces:
                assert isinstance(ns, NamespaceDescription)
                assert isinstance(ns.name, str)
                assert isinstance(ns.record_count, int)
                collected_names.append(ns.name)

            pages_seen += 1

            if page.pagination is not None and page.pagination.next is not None:
                # Intermediate page — token must be a non-empty string
                assert isinstance(page.pagination.next, str)
                assert len(page.pagination.next) > 0
                token = page.pagination.next
            else:
                # Final page — no token (unified-ns-0008)
                break

        # All 3 namespaces found across pages
        for ns in ns_names:
            assert ns in collected_names, (
                f"Expected {ns!r} in paginated results; got {collected_names}"
            )
        # At least 3 pages (one per namespace with limit=1)
        assert pages_seen >= 3, (
            f"Expected >=3 pages with limit=1 and 3 namespaces; saw {pages_seen}"
        )
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )
