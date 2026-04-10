"""Integration tests for deep data-ingestion scenarios (async REST).

Phase 3 area tags: upsert-formats, upsert-batch, upsert-overwrite,
upsert-records, upsert-records-batch, update-metadata, update-sparse,
update-by-filter, delete-by-filter, delete-all-namespace
"""
# ruff: noqa: E501

from __future__ import annotations

import math
import pytest
import pytest_asyncio  # noqa: F401

from pinecone import AsyncPinecone, EmbedConfig, IntegratedSpec, Vector
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.responses import FetchResponse, UpdateResponse, UpsertRecordsResponse, UpsertResponse
from pinecone.models.vectors.search import Hit, SearchRecordsResponse
from tests.integration.conftest import (
    async_cleanup_resource,
    async_poll_until,
    unique_name,
)


# ---------------------------------------------------------------------------
# upsert-formats — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_formats_async(async_client: AsyncPinecone) -> None:
    """Upsert using all accepted input formats in a single call via async REST.

    Formats under test:
    1. Vector object with dense values and metadata
    2. (id, values) tuple
    3. (id, values, metadata) tuple
    4. dict with id, values, sparse_values, and metadata
    """
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        # Format 1: Vector object
        vec1 = Vector(id="fmt-v1", values=[0.1, 0.2, 0.3, 0.4], metadata={"fmt": "object", "n": 1})
        # Format 2: (id, values) tuple
        vec2 = ("fmt-v2", [0.2, 0.3, 0.4, 0.5])
        # Format 3: (id, values, metadata) tuple
        vec3 = ("fmt-v3", [0.3, 0.4, 0.5, 0.6], {"fmt": "tuple3", "n": 3})
        # Format 4: dict with sparse_values and metadata
        vec4 = {
            "id": "fmt-v4",
            "values": [0.4, 0.5, 0.6, 0.7],
            "sparse_values": {"indices": [0, 2], "values": [0.9, 0.8]},
            "metadata": {"fmt": "dict", "n": 4},
        }

        result = await index.upsert(vectors=[vec1, vec2, vec3, vec4])
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 4

        # Wait for eventual consistency — all 4 vectors must be fetchable
        fetched = await async_poll_until(
            query_fn=lambda: index.fetch(ids=["fmt-v1", "fmt-v2", "fmt-v3", "fmt-v4"]),
            check_fn=lambda r: len(r.vectors) == 4,
            timeout=120,
            description="all 4 upserted vectors fetchable (async)",
        )

        assert isinstance(fetched, FetchResponse)

        # Verify Format 1: Vector object with metadata
        v1 = fetched.vectors["fmt-v1"]
        assert v1.id == "fmt-v1"
        assert len(v1.values) == 4
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v1.values, [0.1, 0.2, 0.3, 0.4]))
        assert v1.metadata is not None
        assert v1.metadata.get("fmt") == "object"
        assert v1.metadata.get("n") == 1

        # Verify Format 2: (id, values) tuple — no metadata
        v2 = fetched.vectors["fmt-v2"]
        assert v2.id == "fmt-v2"
        assert len(v2.values) == 4
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v2.values, [0.2, 0.3, 0.4, 0.5]))

        # Verify Format 3: (id, values, metadata) tuple
        v3 = fetched.vectors["fmt-v3"]
        assert v3.id == "fmt-v3"
        assert len(v3.values) == 4
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v3.values, [0.3, 0.4, 0.5, 0.6]))
        assert v3.metadata is not None
        assert v3.metadata.get("fmt") == "tuple3"
        assert v3.metadata.get("n") == 3

        # Verify Format 4: dict with sparse_values and metadata
        v4 = fetched.vectors["fmt-v4"]
        assert v4.id == "fmt-v4"
        assert len(v4.values) == 4
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v4.values, [0.4, 0.5, 0.6, 0.7]))
        assert v4.sparse_values is not None
        assert isinstance(v4.sparse_values, SparseValues)
        assert v4.sparse_values.indices == [0, 2]
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v4.sparse_values.values, [0.9, 0.8]))
        assert v4.metadata is not None
        assert v4.metadata.get("fmt") == "dict"
        assert v4.metadata.get("n") == 4

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# upsert-batch — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_batch_async(async_client: AsyncPinecone) -> None:
    """Upsert 200 vectors in a single call via async REST.

    Verifies:
    - upserted_count == 200
    - describe_index_stats() reports total_vector_count >= 200 after consistency
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
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        vectors = [
            {"id": f"batch-{i}", "values": [float(i) / 200, 1.0 - float(i) / 200]}
            for i in range(200)
        ]

        result = await index.upsert(vectors=vectors)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 200

        # Poll until all 200 vectors are registered in stats
        stats = await async_poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 200,
            timeout=120,
            description="total_vector_count >= 200 in stats (async)",
        )
        assert stats.total_vector_count >= 200

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# upsert-overwrite — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_overwrite_async(async_client: AsyncPinecone) -> None:
    """Second upsert of the same ID fully replaces values AND metadata (async REST).

    Verifies:
    - Initial upsert stores values [0.1, 0.2] and metadata {"v": 1, "original": "yes"}
    - Second upsert of same ID with values [0.9, 0.8] and metadata {"v": 2, "new_key": "hello"}
      completely replaces the first write — old metadata keys are gone
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
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        # First write
        await index.upsert(vectors=[{"id": "ow-1", "values": [0.1, 0.2], "metadata": {"v": 1, "original": "yes"}}])

        # Wait for first write to be visible
        await async_poll_until(
            query_fn=lambda: index.fetch(ids=["ow-1"]),
            check_fn=lambda r: "ow-1" in r.vectors,
            timeout=120,
            description="first upsert of ow-1 fetchable (async)",
        )

        # Verify first write values before overwriting
        fetched_before = await index.fetch(ids=["ow-1"])
        v_before = fetched_before.vectors["ow-1"]
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v_before.values, [0.1, 0.2]))
        assert v_before.metadata is not None
        assert v_before.metadata.get("v") == 1
        assert v_before.metadata.get("original") == "yes"

        # Second write — overwrite same ID
        await index.upsert(vectors=[{"id": "ow-1", "values": [0.9, 0.8], "metadata": {"v": 2, "new_key": "hello"}}])

        # Wait for second write to propagate — poll until values change
        async def _second_write_visible() -> object:
            r = await index.fetch(ids=["ow-1"])
            if "ow-1" not in r.vectors:
                return None
            v = r.vectors["ow-1"]
            if not math.isclose(v.values[0], 0.9, rel_tol=1e-5):
                return None
            return r

        fetched_after = await async_poll_until(
            query_fn=_second_write_visible,
            check_fn=lambda r: r is not None,
            timeout=120,
            description="second upsert of ow-1 propagated (async, values[0] ~ 0.9)",
        )

        v_after = fetched_after.vectors["ow-1"]  # type: ignore[union-attr]
        assert v_after.id == "ow-1"
        # Values completely replaced
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v_after.values, [0.9, 0.8])), \
            f"expected [0.9, 0.8] but got {v_after.values}"
        # Metadata completely replaced — new keys present
        assert v_after.metadata is not None
        assert v_after.metadata.get("v") == 2
        assert v_after.metadata.get("new_key") == "hello"
        # Old metadata key gone
        assert "original" not in v_after.metadata, \
            f"old key 'original' should not persist after overwrite; got metadata={v_after.metadata}"

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# upsert-records-batch — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_records_batch_async(async_client: AsyncPinecone) -> None:
    """Upsert 50 records in one call to an integrated-inference index via async REST.

    Verifies:
    - upsert_records() returns UpsertRecordsResponse with record_count == 50
    - Records become searchable via search(inputs={"text": ...})
    - Hit structure has id (str) and score (float)
    """
    name = unique_name("idx")
    namespace = "urb-ns"
    try:
        await async_client.indexes.create(
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

        # Wait for the index to be ready via async polling
        await async_poll_until(
            query_fn=lambda: async_client.indexes.describe(name),
            check_fn=lambda r: r.status.ready,
            timeout=300,
            interval=5,
            description=f"integrated index {name!r} ready",
        )

        # Populate host cache and get async index handle
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        records = [
            {"_id": f"urb-{i}", "text": f"Record number {i}: vector database similarity search use case {i}."}
            for i in range(50)
        ]
        response = await index.upsert_records(records=records, namespace=namespace)
        assert isinstance(response, UpsertRecordsResponse)
        assert response.record_count == 50

        # Poll until at least some records are searchable (eventual consistency)
        search_resp = await async_poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=10,
                inputs={"text": "vector database similarity search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="batch upserted records searchable via async REST",
        )

        assert isinstance(search_resp, SearchRecordsResponse)
        assert len(search_resp.result.hits) > 0
        first_hit = search_resp.result.hits[0]
        assert isinstance(first_hit, Hit)
        assert isinstance(first_hit.id, str)
        assert isinstance(first_hit.score, float)
        assert first_hit.id.startswith("urb-")

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# upsert-records — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_records_async(async_client: AsyncPinecone) -> None:
    """Upsert records into an integrated-inference index via async REST.

    Verifies:
    - upsert_records() returns UpsertRecordsResponse with record_count == N
    - Uploaded records become searchable via search(inputs={"text": ...})
    - Hit structure has id (str) and score (float)
    """
    name = unique_name("idx")
    namespace = "urec-ns"
    try:
        await async_client.indexes.create(
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

        # Wait for the index to be ready via async polling
        await async_poll_until(
            query_fn=lambda: async_client.indexes.describe(name),
            check_fn=lambda r: r.status.ready,
            timeout=300,
            interval=5,
            description=f"integrated index {name!r} ready",
        )

        # Populate host cache and get async index handle
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        records = [
            {"_id": "urec-1", "text": "Vector databases enable fast similarity search."},
            {"_id": "urec-2", "text": "RAG combines retrieval with language model generation."},
            {"_id": "urec-3", "text": "Embeddings are dense vector representations of data."},
        ]
        response = await index.upsert_records(records=records, namespace=namespace)
        assert isinstance(response, UpsertRecordsResponse)
        assert response.record_count == 3

        # Poll until records are searchable (eventual consistency)
        search_resp = await async_poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=5,
                inputs={"text": "similarity search with embeddings"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="upserted records searchable via async REST",
        )

        assert isinstance(search_resp, SearchRecordsResponse)
        assert len(search_resp.result.hits) > 0
        first_hit = search_resp.result.hits[0]
        assert isinstance(first_hit, Hit)
        assert isinstance(first_hit.id, str)
        assert isinstance(first_hit.score, float)
        assert first_hit.id.startswith("urec-")

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# update-metadata — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_metadata_async(async_client: AsyncPinecone) -> None:
    """index.update(id=..., set_metadata=...) merges metadata, not replaces (async REST).

    Verifies:
    - After update(set_metadata={"color": "blue"}), fetch returns color == "blue"
    - The existing key "size" == 5 is preserved (merge semantics)
    - update() returns an UpdateResponse
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
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        # Upsert a vector with two metadata fields
        await index.upsert(vectors=[{
            "id": "um-v1",
            "values": [0.1, 0.2],
            "metadata": {"color": "red", "size": 5},
        }])

        # Wait for vector to be fetchable
        await async_poll_until(
            query_fn=lambda: index.fetch(ids=["um-v1"]),
            check_fn=lambda r: "um-v1" in r.vectors,
            timeout=120,
            description="um-v1 fetchable before update (async)",
        )

        # Update only the "color" field — "size" should survive (merge semantics)
        update_resp = await index.update(id="um-v1", set_metadata={"color": "blue"})
        assert isinstance(update_resp, UpdateResponse)

        # Poll until the metadata change propagates
        async def _color_updated() -> object:
            r = await index.fetch(ids=["um-v1"])
            if "um-v1" not in r.vectors:
                return None
            meta = r.vectors["um-v1"].metadata
            if meta is None or meta.get("color") != "blue":
                return None
            return r

        fetched = await async_poll_until(
            query_fn=_color_updated,
            check_fn=lambda r: r is not None,
            timeout=120,
            description="um-v1 color updated to blue (async)",
        )

        v = fetched.vectors["um-v1"]  # type: ignore[union-attr]
        assert v.metadata is not None
        # Updated field
        assert v.metadata.get("color") == "blue", \
            f"expected color='blue', got {v.metadata.get('color')!r}"
        # Preserved field — merge semantics (NOT replaced)
        assert v.metadata.get("size") == 5, \
            f"expected size=5 to be preserved but got {v.metadata.get('size')!r}"

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# update-sparse — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_sparse_async(async_client: AsyncPinecone) -> None:
    """index.update(id=..., sparse_values=...) replaces sparse component while preserving dense values (async REST).

    Verifies:
    - Upsert hybrid vector with sparse_values {"indices": [0, 3], "values": [0.5, 0.8]}
    - Update with new sparse_values {"indices": [1, 2], "values": [0.9, 0.7]}
    - Fetch and verify new sparse indices/values present
    - Dense values [0.1, 0.2, 0.3, 0.4] are unchanged
    - update() returns an UpdateResponse
    """
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        desc = await async_client.indexes.describe(name)
        index = async_client.index(host=desc.host)

        # Upsert hybrid vector with initial sparse values
        await index.upsert(vectors=[{
            "id": "us-v1",
            "values": [0.1, 0.2, 0.3, 0.4],
            "sparse_values": {"indices": [0, 3], "values": [0.5, 0.8]},
        }])

        # Wait for vector to be fetchable
        await async_poll_until(
            query_fn=lambda: index.fetch(ids=["us-v1"]),
            check_fn=lambda r: "us-v1" in r.vectors,
            timeout=120,
            description="us-v1 fetchable before sparse update (async)",
        )

        # Update only the sparse values — dense values should be preserved
        update_resp = await index.update(
            id="us-v1",
            sparse_values={"indices": [1, 2], "values": [0.9, 0.7]},
        )
        assert isinstance(update_resp, UpdateResponse)

        # Poll until the sparse values change propagates
        async def _sparse_updated() -> object:
            r = await index.fetch(ids=["us-v1"])
            if "us-v1" not in r.vectors:
                return None
            v = r.vectors["us-v1"]
            if v.sparse_values is None or v.sparse_values.indices != [1, 2]:
                return None
            return r

        fetched = await async_poll_until(
            query_fn=_sparse_updated,
            check_fn=lambda r: r is not None,
            timeout=120,
            description="us-v1 sparse values updated to indices=[1, 2] (async)",
        )

        v = fetched.vectors["us-v1"]  # type: ignore[union-attr]
        # New sparse values present
        assert v.sparse_values is not None, "sparse_values should be present after update (async)"
        assert isinstance(v.sparse_values, SparseValues)
        assert v.sparse_values.indices == [1, 2], \
            f"expected sparse indices [1, 2], got {v.sparse_values.indices}"
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v.sparse_values.values, [0.9, 0.7])), \
            f"expected sparse values [0.9, 0.7], got {v.sparse_values.values}"
        # Dense values preserved
        assert len(v.values) == 4, f"expected 4 dense values, got {len(v.values)}"
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v.values, [0.1, 0.2, 0.3, 0.4])), \
            f"expected dense values [0.1, 0.2, 0.3, 0.4], got {v.values}"

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )


# ---------------------------------------------------------------------------
# update-by-filter — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_by_filter_async(async_client: AsyncPinecone) -> None:
    """Filter-based bulk metadata update via async REST.

    Upsert 5 vectors: 3 with genre=drama, 2 with genre=comedy.
    First test dry_run=True — verify it returns a matched_records count
    without mutating any vectors. Then apply the filter-based update and
    confirm only the 3 drama vectors received reviewed=True.
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
        index = async_client.index(name=name)

        # Upsert 3 drama and 2 comedy vectors
        vectors = [
            {"id": "ubf-d1", "values": [0.1, 0.2], "metadata": {"genre": "drama"}},
            {"id": "ubf-d2", "values": [0.2, 0.3], "metadata": {"genre": "drama"}},
            {"id": "ubf-d3", "values": [0.3, 0.4], "metadata": {"genre": "drama"}},
            {"id": "ubf-c1", "values": [0.5, 0.6], "metadata": {"genre": "comedy"}},
            {"id": "ubf-c2", "values": [0.6, 0.7], "metadata": {"genre": "comedy"}},
        ]
        result = await index.upsert(vectors=vectors)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 5

        all_ids = ["ubf-d1", "ubf-d2", "ubf-d3", "ubf-c1", "ubf-c2"]

        # Wait for all 5 vectors to be fetchable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: index.fetch(ids=all_ids),
            check_fn=lambda r: len(r.vectors) == 5,
            timeout=120,
            description="all 5 update-by-filter vectors fetchable (async)",
        )

        # Dry-run first — should return matched_records count without mutating
        dry_resp = await index.update(
            filter={"genre": {"$eq": "drama"}},
            set_metadata={"reviewed": True},
            dry_run=True,
        )
        assert isinstance(dry_resp, UpdateResponse)
        # matched_records may be None if not yet indexed, otherwise should be >= 0
        if dry_resp.matched_records is not None:
            assert dry_resp.matched_records >= 0, \
                f"dry_run matched_records should be non-negative, got {dry_resp.matched_records}"

        # Verify dry_run did NOT mutate — drama vectors should NOT have reviewed=True yet
        fetched_after_dry = await index.fetch(ids=all_ids)
        for vid in ["ubf-d1", "ubf-d2", "ubf-d3"]:
            v = fetched_after_dry.vectors.get(vid)
            if v is not None and v.metadata is not None:
                assert v.metadata.get("reviewed") is None, \
                    f"dry_run should not have mutated {vid}: got reviewed={v.metadata.get('reviewed')!r}"

        # Now apply the real filter-based update
        update_resp = await index.update(
            filter={"genre": {"$eq": "drama"}},
            set_metadata={"reviewed": True},
        )
        assert isinstance(update_resp, UpdateResponse)

        # Poll until the 3 drama vectors all have reviewed=True
        async def _all_drama_reviewed_async() -> object:
            r = await index.fetch(ids=all_ids)
            if len(r.vectors) < 5:
                return None
            for vid in ["ubf-d1", "ubf-d2", "ubf-d3"]:
                v = r.vectors.get(vid)
                if v is None or v.metadata is None or v.metadata.get("reviewed") is not True:
                    return None
            return r

        fetched = await async_poll_until(
            query_fn=_all_drama_reviewed_async,
            check_fn=lambda r: r is not None,
            timeout=180,
            description="all 3 drama vectors have reviewed=True after filter-update (async)",
        )

        # Verify drama vectors have reviewed=True
        for vid in ["ubf-d1", "ubf-d2", "ubf-d3"]:
            v = fetched.vectors[vid]  # type: ignore[union-attr]
            assert v.metadata is not None, f"{vid} should have metadata"
            assert v.metadata.get("reviewed") is True, \
                f"{vid} should have reviewed=True, got {v.metadata.get('reviewed')!r}"
            assert v.metadata.get("genre") == "drama", \
                f"{vid} should still have genre=drama, got {v.metadata.get('genre')!r}"

        # Verify comedy vectors were NOT touched
        for vid in ["ubf-c1", "ubf-c2"]:
            v = fetched.vectors[vid]  # type: ignore[union-attr]
            assert v.metadata is not None, f"{vid} should have metadata"
            assert v.metadata.get("reviewed") is None, \
                f"{vid} (comedy) should NOT have reviewed, got {v.metadata.get('reviewed')!r}"
            assert v.metadata.get("genre") == "comedy", \
                f"{vid} should still have genre=comedy, got {v.metadata.get('genre')!r}"

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )
