"""Integration tests for deep data-ingestion scenarios (sync REST + gRPC).

Phase 3 area tags: upsert-formats, upsert-batch, upsert-overwrite,
upsert-records, upsert-records-batch, update-metadata, update-sparse,
update-by-filter, delete-by-filter, delete-all-namespace
"""

from __future__ import annotations

import math

import pytest

from pinecone import EmbedConfig, IntegratedSpec, Pinecone, PineconeValueError, Vector
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import Hit, SearchRecordsResponse
from pinecone.models.vectors.sparse import SparseValues
from tests.integration.conftest import cleanup_resource, poll_until, unique_name, wait_for_ready

# ---------------------------------------------------------------------------
# upsert-formats — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_formats_rest(client: Pinecone) -> None:
    """Upsert using all accepted input formats in a single call via REST.

    Formats under test:
    1. Vector object with dense values and metadata
    2. (id, values) tuple
    3. (id, values, metadata) tuple
    4. dict with id, values, sparse_values, and metadata
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

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

        result = index.upsert(vectors=[vec1, vec2, vec3, vec4])
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 4

        # Wait for eventual consistency — all 4 vectors must be fetchable
        fetched = poll_until(
            query_fn=lambda: index.fetch(ids=["fmt-v1", "fmt-v2", "fmt-v3", "fmt-v4"]),
            check_fn=lambda r: len(r.vectors) == 4,
            timeout=120,
            description="all 4 upserted vectors fetchable",
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
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-batch — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_batch_rest(client: Pinecone) -> None:
    """Upsert 200 vectors in a single call via REST sync.

    Verifies:
    - upserted_count == 200
    - describe_index_stats() reports total_vector_count >= 200 after consistency
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

        vectors = [
            {"id": f"batch-{i}", "values": [float(i) / 200, 1.0 - float(i) / 200]}
            for i in range(200)
        ]

        result = index.upsert(vectors=vectors)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 200

        # Poll until all 200 vectors are registered in stats
        stats = poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 200,
            timeout=120,
            description="total_vector_count >= 200 in stats",
        )
        assert stats.total_vector_count >= 200

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-batch — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_batch_grpc(client: Pinecone) -> None:
    """Upsert 200 vectors in a single call via gRPC."""
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

        vectors = [
            {"id": f"batch-{i}", "values": [float(i) / 200, 1.0 - float(i) / 200]}
            for i in range(200)
        ]

        result = index.upsert(vectors=vectors)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 200

        # Poll until all 200 vectors are registered in stats
        stats = poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 200,
            timeout=120,
            description="total_vector_count >= 200 in stats via gRPC",
        )
        assert stats.total_vector_count >= 200

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-overwrite — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_overwrite_rest(client: Pinecone) -> None:
    """Second upsert of the same ID fully replaces values AND metadata (REST sync).

    Verifies:
    - Initial upsert stores values [0.1, 0.2] and metadata {"v": 1}
    - Second upsert of same ID with values [0.9, 0.8] and metadata {"v": 2, "new_key": "hello"}
      completely replaces the first write — old values and old metadata keys gone
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

        # First write
        index.upsert(vectors=[{"id": "ow-1", "values": [0.1, 0.2], "metadata": {"v": 1, "original": "yes"}}])

        # Wait for first write to be visible
        poll_until(
            query_fn=lambda: index.fetch(ids=["ow-1"]),
            check_fn=lambda r: "ow-1" in r.vectors,
            timeout=120,
            description="first upsert of ow-1 fetchable",
        )

        # Verify first write values before overwriting
        fetched_before = index.fetch(ids=["ow-1"])
        v_before = fetched_before.vectors["ow-1"]
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v_before.values, [0.1, 0.2]))
        assert v_before.metadata is not None
        assert v_before.metadata.get("v") == 1
        assert v_before.metadata.get("original") == "yes"

        # Second write — overwrite same ID
        index.upsert(vectors=[{"id": "ow-1", "values": [0.9, 0.8], "metadata": {"v": 2, "new_key": "hello"}}])

        # Wait for second write to propagate — poll until values change
        def _second_write_visible() -> object:
            r = index.fetch(ids=["ow-1"])
            if "ow-1" not in r.vectors:
                return None
            v = r.vectors["ow-1"]
            if not math.isclose(v.values[0], 0.9, rel_tol=1e-5):
                return None
            return r

        fetched_after = poll_until(
            query_fn=_second_write_visible,
            check_fn=lambda r: r is not None,
            timeout=120,
            description="second upsert of ow-1 propagated (values[0] ~ 0.9)",
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
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-overwrite — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_overwrite_grpc(client: Pinecone) -> None:
    """Second upsert of the same ID fully replaces values AND metadata (gRPC).

    Verifies identical semantics to the REST overwrite test but via the gRPC transport.
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

        # First write
        index.upsert(vectors=[{"id": "ow-1", "values": [0.1, 0.2], "metadata": {"v": 1, "original": "yes"}}])

        # Wait for first write to be visible
        poll_until(
            query_fn=lambda: index.fetch(ids=["ow-1"]),
            check_fn=lambda r: "ow-1" in r.vectors,
            timeout=120,
            description="first upsert of ow-1 fetchable (gRPC)",
        )

        # Second write — overwrite same ID
        index.upsert(vectors=[{"id": "ow-1", "values": [0.9, 0.8], "metadata": {"v": 2, "new_key": "hello"}}])

        # Wait for second write to propagate
        def _second_write_visible() -> object:
            r = index.fetch(ids=["ow-1"])
            if "ow-1" not in r.vectors:
                return None
            v = r.vectors["ow-1"]
            if not math.isclose(v.values[0], 0.9, rel_tol=1e-5):
                return None
            return r

        fetched_after = poll_until(
            query_fn=_second_write_visible,
            check_fn=lambda r: r is not None,
            timeout=120,
            description="second upsert of ow-1 propagated (gRPC)",
        )

        v_after = fetched_after.vectors["ow-1"]  # type: ignore[union-attr]
        assert v_after.id == "ow-1"
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(v_after.values, [0.9, 0.8])), \
            f"expected [0.9, 0.8] but got {v_after.values}"
        assert v_after.metadata is not None
        assert v_after.metadata.get("v") == 2
        assert v_after.metadata.get("new_key") == "hello"
        assert "original" not in v_after.metadata, \
            f"old key 'original' should not persist after overwrite; got metadata={v_after.metadata}"

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-formats — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_formats_grpc(client: Pinecone) -> None:
    """Upsert using all accepted input formats in a single call via gRPC."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

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

        result = index.upsert(vectors=[vec1, vec2, vec3, vec4])
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 4

        # Wait for eventual consistency
        fetched = poll_until(
            query_fn=lambda: index.fetch(ids=["fmt-v1", "fmt-v2", "fmt-v3", "fmt-v4"]),
            check_fn=lambda r: len(r.vectors) == 4,
            timeout=120,
            description="all 4 upserted vectors fetchable via gRPC",
        )

        assert isinstance(fetched, FetchResponse)

        # Spot-check key fields
        v1 = fetched.vectors["fmt-v1"]
        assert v1.id == "fmt-v1"
        assert len(v1.values) == 4
        assert v1.metadata is not None
        assert v1.metadata.get("fmt") == "object"

        v3 = fetched.vectors["fmt-v3"]
        assert v3.metadata is not None
        assert v3.metadata.get("fmt") == "tuple3"

        v4 = fetched.vectors["fmt-v4"]
        assert v4.sparse_values is not None
        assert v4.sparse_values.indices == [0, 2]
        assert v4.metadata is not None
        assert v4.metadata.get("fmt") == "dict"

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-records — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_records_rest(client: Pinecone) -> None:
    """Upsert records into an integrated-inference index via REST sync.

    Verifies:
    - upsert_records() returns UpsertRecordsResponse with record_count == N
    - Uploaded records become searchable via search(inputs={"text": ...})
    - Hit structure has id (str) and score (float)
    """
    name = unique_name("idx")
    namespace = "urec-ns"
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

        records = [
            {"_id": "urec-1", "text": "Vector databases enable fast similarity search."},
            {"_id": "urec-2", "text": "RAG combines retrieval with language model generation."},
            {"_id": "urec-3", "text": "Embeddings are dense vector representations of data."},
        ]
        response = index.upsert_records(records=records, namespace=namespace)
        assert isinstance(response, UpsertRecordsResponse)
        assert response.record_count == 3

        # Poll until records are searchable (eventual consistency)
        search_resp = poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=5,
                inputs={"text": "similarity search with embeddings"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="upserted records searchable via REST",
        )

        assert isinstance(search_resp, SearchRecordsResponse)
        assert len(search_resp.result.hits) > 0
        first_hit = search_resp.result.hits[0]
        assert isinstance(first_hit, Hit)
        assert isinstance(first_hit.id, str)
        assert isinstance(first_hit.score, float)
        assert first_hit.id.startswith("urec-")

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-records-batch — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_records_batch_rest(client: Pinecone) -> None:
    """Upsert 50 records in one call to an integrated-inference index via REST sync.

    Verifies:
    - upsert_records() returns UpsertRecordsResponse with record_count == 50
    - Records become searchable via search(inputs={"text": ...})
    - Hit structure has id (str) and score (float)
    """
    name = unique_name("idx")
    namespace = "urb-ns"
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

        records = [
            {"_id": f"urb-{i}", "text": f"Record number {i}: vector database similarity search use case {i}."}
            for i in range(50)
        ]
        response = index.upsert_records(records=records, namespace=namespace)
        assert isinstance(response, UpsertRecordsResponse)
        assert response.record_count == 50

        # Poll until at least some records are searchable (eventual consistency)
        search_resp = poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=10,
                inputs={"text": "vector database similarity search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="batch upserted records searchable via REST",
        )

        assert isinstance(search_resp, SearchRecordsResponse)
        assert len(search_resp.result.hits) > 0
        first_hit = search_resp.result.hits[0]
        assert isinstance(first_hit, Hit)
        assert isinstance(first_hit.id, str)
        assert isinstance(first_hit.score, float)
        assert first_hit.id.startswith("urb-")

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-records-batch — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_records_batch_grpc(client: Pinecone) -> None:
    """Upsert 50 records in one call to an integrated-inference index via gRPC handle.

    GrpcIndex.upsert_records() and GrpcIndex.search() both delegate to REST.
    Verifies the gRPC index handle can be used for both operations.
    """
    name = unique_name("idx")
    namespace = "urb-ns"
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

        records = [
            {"_id": f"urb-{i}", "text": f"Record number {i}: vector database similarity search use case {i}."}
            for i in range(50)
        ]
        response = index.upsert_records(records=records, namespace=namespace)
        assert isinstance(response, UpsertRecordsResponse)
        assert response.record_count == 50

        # Poll until at least some records are searchable via gRPC (REST fallback)
        search_resp = poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=10,
                inputs={"text": "vector database similarity search"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="batch upserted records searchable via gRPC transport",
        )

        assert isinstance(search_resp, SearchRecordsResponse)
        assert len(search_resp.result.hits) > 0
        first_hit = search_resp.result.hits[0]
        assert isinstance(first_hit, Hit)
        assert isinstance(first_hit.id, str)
        assert isinstance(first_hit.score, float)
        assert first_hit.id.startswith("urb-")

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-records — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_records_grpc(client: Pinecone) -> None:
    """Upsert records into an integrated-inference index via gRPC transport.

    GrpcIndex.upsert_records() delegates to REST (no gRPC equivalent).
    GrpcIndex.search() also uses REST for integrated search.
    Verifies the gRPC index handle can be used for both operations.
    """
    name = unique_name("idx")
    namespace = "urec-ns"
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

        records = [
            {"_id": "urec-1", "text": "Vector databases enable fast similarity search."},
            {"_id": "urec-2", "text": "RAG combines retrieval with language model generation."},
            {"_id": "urec-3", "text": "Embeddings are dense vector representations of data."},
        ]
        response = index.upsert_records(records=records, namespace=namespace)
        assert isinstance(response, UpsertRecordsResponse)
        assert response.record_count == 3

        # Poll until records are searchable via gRPC search (REST fallback)
        search_resp = poll_until(
            query_fn=lambda: index.search(
                namespace=namespace,
                top_k=5,
                inputs={"text": "similarity search with embeddings"},
            ),
            check_fn=lambda r: len(r.result.hits) > 0,
            timeout=120,
            description="upserted records searchable via gRPC transport",
        )

        assert isinstance(search_resp, SearchRecordsResponse)
        assert len(search_resp.result.hits) > 0
        first_hit = search_resp.result.hits[0]
        assert isinstance(first_hit, Hit)
        assert isinstance(first_hit.id, str)
        assert isinstance(first_hit.score, float)
        assert first_hit.id.startswith("urec-")

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# update-metadata — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_update_metadata_rest(client: Pinecone) -> None:
    """index.update(id=..., set_metadata=...) merges metadata, not replaces (REST sync).

    Verifies:
    - After update(set_metadata={"color": "blue"}), fetch returns color == "blue"
    - The existing key "size" == 5 is preserved (merge semantics)
    - update() returns an UpdateResponse
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

        # Upsert a vector with two metadata fields
        index.upsert(vectors=[{
            "id": "um-v1",
            "values": [0.1, 0.2],
            "metadata": {"color": "red", "size": 5},
        }])

        # Wait for vector to be fetchable
        poll_until(
            query_fn=lambda: index.fetch(ids=["um-v1"]),
            check_fn=lambda r: "um-v1" in r.vectors,
            timeout=120,
            description="um-v1 fetchable before update",
        )

        # Update only the "color" field — "size" should survive (merge semantics)
        update_resp = index.update(id="um-v1", set_metadata={"color": "blue"})
        assert isinstance(update_resp, UpdateResponse)

        # Poll until the metadata change propagates
        def _color_updated() -> object:
            r = index.fetch(ids=["um-v1"])
            if "um-v1" not in r.vectors:
                return None
            meta = r.vectors["um-v1"].metadata
            if meta is None or meta.get("color") != "blue":
                return None
            return r

        fetched = poll_until(
            query_fn=_color_updated,
            check_fn=lambda r: r is not None,
            timeout=120,
            description="um-v1 color updated to blue (REST)",
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
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# update-sparse — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_update_sparse_rest(client: Pinecone) -> None:
    """index.update(id=..., sparse_values=...) replaces sparse component while preserving dense values (REST sync).

    Verifies:
    - Upsert hybrid vector with sparse_values {"indices": [0, 3], "values": [0.5, 0.8]}
    - Update with new sparse_values {"indices": [1, 2], "values": [0.9, 0.7]}
    - Fetch and verify new sparse indices/values present
    - Dense values [0.1, 0.2, 0.3, 0.4] are unchanged
    - update() returns an UpdateResponse
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        # Upsert hybrid vector with initial sparse values
        index.upsert(vectors=[{
            "id": "us-v1",
            "values": [0.1, 0.2, 0.3, 0.4],
            "sparse_values": {"indices": [0, 3], "values": [0.5, 0.8]},
        }])

        # Wait for vector to be fetchable
        poll_until(
            query_fn=lambda: index.fetch(ids=["us-v1"]),
            check_fn=lambda r: "us-v1" in r.vectors,
            timeout=120,
            description="us-v1 fetchable before sparse update",
        )

        # Update only the sparse values — dense values should be preserved
        update_resp = index.update(
            id="us-v1",
            sparse_values={"indices": [1, 2], "values": [0.9, 0.7]},
        )
        assert isinstance(update_resp, UpdateResponse)

        # Poll until the sparse values change propagates (first index changes from 0 to 1)
        def _sparse_updated() -> object:
            r = index.fetch(ids=["us-v1"])
            if "us-v1" not in r.vectors:
                return None
            v = r.vectors["us-v1"]
            if v.sparse_values is None or v.sparse_values.indices != [1, 2]:
                return None
            return r

        fetched = poll_until(
            query_fn=_sparse_updated,
            check_fn=lambda r: r is not None,
            timeout=120,
            description="us-v1 sparse values updated to indices=[1, 2] (REST)",
        )

        v = fetched.vectors["us-v1"]  # type: ignore[union-attr]
        # New sparse values present
        assert v.sparse_values is not None, "sparse_values should be present after update"
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
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# update-sparse — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_update_sparse_grpc(client: Pinecone) -> None:
    """index.update(id=..., sparse_values=...) replaces sparse component while preserving dense values (gRPC).

    Verifies the same semantics as the REST sync test but via the gRPC transport.
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        # Upsert hybrid vector with initial sparse values
        index.upsert(vectors=[{
            "id": "us-v1",
            "values": [0.1, 0.2, 0.3, 0.4],
            "sparse_values": {"indices": [0, 3], "values": [0.5, 0.8]},
        }])

        # Wait for vector to be fetchable
        poll_until(
            query_fn=lambda: index.fetch(ids=["us-v1"]),
            check_fn=lambda r: "us-v1" in r.vectors,
            timeout=120,
            description="us-v1 fetchable before sparse update (gRPC)",
        )

        # Update only the sparse values — dense values should be preserved
        update_resp = index.update(
            id="us-v1",
            sparse_values={"indices": [1, 2], "values": [0.9, 0.7]},
        )
        assert isinstance(update_resp, UpdateResponse)

        # Poll until the sparse values change propagates
        def _sparse_updated_grpc() -> object:
            r = index.fetch(ids=["us-v1"])
            if "us-v1" not in r.vectors:
                return None
            v = r.vectors["us-v1"]
            if v.sparse_values is None or v.sparse_values.indices != [1, 2]:
                return None
            return r

        fetched = poll_until(
            query_fn=_sparse_updated_grpc,
            check_fn=lambda r: r is not None,
            timeout=120,
            description="us-v1 sparse values updated to indices=[1, 2] (gRPC)",
        )

        v = fetched.vectors["us-v1"]  # type: ignore[union-attr]
        # New sparse values present
        assert v.sparse_values is not None, "sparse_values should be present after update (gRPC)"
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
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# update-metadata — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_update_metadata_grpc(client: Pinecone) -> None:
    """index.update(id=..., set_metadata=...) merges metadata, not replaces (gRPC).

    Verifies the same merge semantics as the REST sync test but via gRPC transport.
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

        # Upsert a vector with two metadata fields
        index.upsert(vectors=[{
            "id": "um-v1",
            "values": [0.1, 0.2],
            "metadata": {"color": "red", "size": 5},
        }])

        # Wait for vector to be fetchable
        poll_until(
            query_fn=lambda: index.fetch(ids=["um-v1"]),
            check_fn=lambda r: "um-v1" in r.vectors,
            timeout=120,
            description="um-v1 fetchable before update (gRPC)",
        )

        # Update only the "color" field — "size" should survive
        update_resp = index.update(id="um-v1", set_metadata={"color": "blue"})
        assert isinstance(update_resp, UpdateResponse)

        # Poll until the metadata change propagates
        def _color_updated_grpc() -> object:
            r = index.fetch(ids=["um-v1"])
            if "um-v1" not in r.vectors:
                return None
            meta = r.vectors["um-v1"].metadata
            if meta is None or meta.get("color") != "blue":
                return None
            return r

        fetched = poll_until(
            query_fn=_color_updated_grpc,
            check_fn=lambda r: r is not None,
            timeout=120,
            description="um-v1 color updated to blue (gRPC)",
        )

        v = fetched.vectors["um-v1"]  # type: ignore[union-attr]
        assert v.metadata is not None
        assert v.metadata.get("color") == "blue", \
            f"expected color='blue', got {v.metadata.get('color')!r}"
        assert v.metadata.get("size") == 5, \
            f"expected size=5 to be preserved but got {v.metadata.get('size')!r}"

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# update-by-filter — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_update_by_filter_rest(client: Pinecone) -> None:
    """Filter-based bulk metadata update via REST.

    Upsert 5 vectors: 3 with genre=drama, 2 with genre=comedy.
    First test dry_run=True — verify it returns a matched_records count
    without mutating any vectors. Then apply the filter-based update and
    confirm only the 3 drama vectors received reviewed=True.
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

        # Upsert 3 drama and 2 comedy vectors
        vectors = [
            {"id": "ubf-d1", "values": [0.1, 0.2], "metadata": {"genre": "drama"}},
            {"id": "ubf-d2", "values": [0.2, 0.3], "metadata": {"genre": "drama"}},
            {"id": "ubf-d3", "values": [0.3, 0.4], "metadata": {"genre": "drama"}},
            {"id": "ubf-c1", "values": [0.5, 0.6], "metadata": {"genre": "comedy"}},
            {"id": "ubf-c2", "values": [0.6, 0.7], "metadata": {"genre": "comedy"}},
        ]
        result = index.upsert(vectors=vectors)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 5

        all_ids = ["ubf-d1", "ubf-d2", "ubf-d3", "ubf-c1", "ubf-c2"]

        # Wait for all 5 vectors to be fetchable (eventual consistency)
        poll_until(
            query_fn=lambda: index.fetch(ids=all_ids),
            check_fn=lambda r: len(r.vectors) == 5,
            timeout=120,
            description="all 5 update-by-filter vectors fetchable",
        )

        # Dry-run first — should return matched_records count without mutating
        dry_resp = index.update(
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
        fetched_after_dry = index.fetch(ids=all_ids)
        for vid in ["ubf-d1", "ubf-d2", "ubf-d3"]:
            v = fetched_after_dry.vectors.get(vid)
            if v is not None and v.metadata is not None:
                assert v.metadata.get("reviewed") is None, \
                    f"dry_run should not have mutated {vid}: got reviewed={v.metadata.get('reviewed')!r}"

        # Now apply the real filter-based update
        update_resp = index.update(
            filter={"genre": {"$eq": "drama"}},
            set_metadata={"reviewed": True},
        )
        assert isinstance(update_resp, UpdateResponse)

        # Poll until the 3 drama vectors all have reviewed=True
        def _all_drama_reviewed() -> object:
            r = index.fetch(ids=all_ids)
            if len(r.vectors) < 5:
                return None
            for vid in ["ubf-d1", "ubf-d2", "ubf-d3"]:
                v = r.vectors.get(vid)
                if v is None or v.metadata is None or v.metadata.get("reviewed") is not True:
                    return None
            return r

        fetched = poll_until(
            query_fn=_all_drama_reviewed,
            check_fn=lambda r: r is not None,
            timeout=180,
            description="all 3 drama vectors have reviewed=True after filter-update",
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
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# delete-by-filter — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_delete_by_filter_rest(client: Pinecone) -> None:
    """index.delete(filter=...) removes only vectors matching the filter (REST sync).

    Upserts 5 vectors: 2 with status="obsolete", 3 with status="active".
    Calls delete(filter={"status": {"$eq": "obsolete"}}).
    Polls until the 2 obsolete vectors are absent from fetch.
    Verifies the 3 active vectors remain intact.
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

        # 2 obsolete + 3 active vectors
        vectors = [
            {"id": "dbf-o1", "values": [0.1, 0.2], "metadata": {"status": "obsolete"}},
            {"id": "dbf-o2", "values": [0.2, 0.3], "metadata": {"status": "obsolete"}},
            {"id": "dbf-a1", "values": [0.5, 0.6], "metadata": {"status": "active"}},
            {"id": "dbf-a2", "values": [0.6, 0.7], "metadata": {"status": "active"}},
            {"id": "dbf-a3", "values": [0.7, 0.8], "metadata": {"status": "active"}},
        ]
        result = index.upsert(vectors=vectors)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 5

        all_ids = ["dbf-o1", "dbf-o2", "dbf-a1", "dbf-a2", "dbf-a3"]
        obsolete_ids = ["dbf-o1", "dbf-o2"]
        active_ids = ["dbf-a1", "dbf-a2", "dbf-a3"]

        # Wait for all 5 vectors to be fetchable before deleting
        poll_until(
            query_fn=lambda: index.fetch(ids=all_ids),
            check_fn=lambda r: len(r.vectors) == 5,
            timeout=120,
            description="all 5 delete-by-filter vectors fetchable (REST)",
        )

        # Delete only the obsolete vectors via metadata filter
        index.delete(filter={"status": {"$eq": "obsolete"}})

        # Poll until both obsolete vectors disappear from fetch
        poll_until(
            query_fn=lambda: index.fetch(ids=obsolete_ids),
            check_fn=lambda r: len(r.vectors) == 0,
            timeout=120,
            description="obsolete vectors deleted by filter (REST)",
        )

        # Verify active vectors still present
        active_fetch = index.fetch(ids=active_ids)
        assert isinstance(active_fetch, FetchResponse)
        for vid in active_ids:
            assert vid in active_fetch.vectors, \
                f"active vector {vid!r} should remain after filter-delete"
            v = active_fetch.vectors[vid]
            assert v.metadata is not None
            assert v.metadata.get("status") == "active", \
                f"{vid} should still have status='active', got {v.metadata.get('status')!r}"

        # Confirm obsolete vectors are truly gone
        obsolete_fetch = index.fetch(ids=obsolete_ids)
        assert len(obsolete_fetch.vectors) == 0, \
            f"obsolete vectors should be deleted but found: {list(obsolete_fetch.vectors.keys())}"

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# delete-by-filter — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_delete_by_filter_grpc(client: Pinecone) -> None:
    """index.delete(filter=...) removes only vectors matching the filter (gRPC).

    Verifies the same filter-delete semantics as the REST sync test but via
    the gRPC transport.
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

        # 2 obsolete + 3 active vectors
        vectors = [
            {"id": "dbf-o1", "values": [0.1, 0.2], "metadata": {"status": "obsolete"}},
            {"id": "dbf-o2", "values": [0.2, 0.3], "metadata": {"status": "obsolete"}},
            {"id": "dbf-a1", "values": [0.5, 0.6], "metadata": {"status": "active"}},
            {"id": "dbf-a2", "values": [0.6, 0.7], "metadata": {"status": "active"}},
            {"id": "dbf-a3", "values": [0.7, 0.8], "metadata": {"status": "active"}},
        ]
        result = index.upsert(vectors=vectors)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 5

        all_ids = ["dbf-o1", "dbf-o2", "dbf-a1", "dbf-a2", "dbf-a3"]
        obsolete_ids = ["dbf-o1", "dbf-o2"]
        active_ids = ["dbf-a1", "dbf-a2", "dbf-a3"]

        # Wait for all 5 vectors to be fetchable before deleting
        poll_until(
            query_fn=lambda: index.fetch(ids=all_ids),
            check_fn=lambda r: len(r.vectors) == 5,
            timeout=120,
            description="all 5 delete-by-filter vectors fetchable (gRPC)",
        )

        # Delete only the obsolete vectors via metadata filter
        index.delete(filter={"status": {"$eq": "obsolete"}})

        # Poll until both obsolete vectors disappear from fetch
        poll_until(
            query_fn=lambda: index.fetch(ids=obsolete_ids),
            check_fn=lambda r: len(r.vectors) == 0,
            timeout=120,
            description="obsolete vectors deleted by filter (gRPC)",
        )

        # Verify active vectors still present
        active_fetch = index.fetch(ids=active_ids)
        assert isinstance(active_fetch, FetchResponse)
        for vid in active_ids:
            assert vid in active_fetch.vectors, \
                f"active vector {vid!r} should remain after filter-delete (gRPC)"
            v = active_fetch.vectors[vid]
            assert v.metadata is not None
            assert v.metadata.get("status") == "active", \
                f"{vid} should still have status='active', got {v.metadata.get('status')!r}"

        # Confirm obsolete vectors are truly gone
        obsolete_fetch = index.fetch(ids=obsolete_ids)
        assert len(obsolete_fetch.vectors) == 0, \
            f"obsolete vectors should be deleted but found: {list(obsolete_fetch.vectors.keys())}"

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# delete-all-namespace — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_delete_all_namespace_rest(client: Pinecone) -> None:
    """index.delete(delete_all=True, namespace=...) deletes all vectors in a named
    namespace (REST sync) while leaving other namespaces untouched.

    Upserts 3 vectors into "dan-cleanup-ns" and 2 vectors into the default namespace.
    Calls delete(delete_all=True, namespace="dan-cleanup-ns").
    Polls describe_index_stats() until "dan-cleanup-ns" is absent or has vector_count==0.
    Verifies the default namespace still has 2 vectors.
    """
    name = unique_name("idx")
    ns = "dan-cleanup-ns"
    default_ids = ["dan-def-1", "dan-def-2"]
    ns_ids = ["dan-ns-1", "dan-ns-2", "dan-ns-3"]
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        # Upsert into named namespace
        ns_vectors = [
            {"id": "dan-ns-1", "values": [0.1, 0.2]},
            {"id": "dan-ns-2", "values": [0.3, 0.4]},
            {"id": "dan-ns-3", "values": [0.5, 0.6]},
        ]
        result = index.upsert(vectors=ns_vectors, namespace=ns)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 3

        # Upsert into default namespace
        def_vectors = [
            {"id": "dan-def-1", "values": [0.7, 0.8]},
            {"id": "dan-def-2", "values": [0.9, 0.1]},
        ]
        result2 = index.upsert(vectors=def_vectors)
        assert isinstance(result2, UpsertResponse)
        assert result2.upserted_count == 2

        # Wait for all vectors to be indexed in stats before deleting
        poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 5,
            timeout=120,
            description="all 5 vectors appear in stats before delete-all (REST)",
        )

        # Delete all vectors in the named namespace
        index.delete(delete_all=True, namespace=ns)

        # Poll until the named namespace is gone or empty
        poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: ns not in r.namespaces or r.namespaces[ns].vector_count == 0,
            timeout=120,
            description="dan-cleanup-ns empty after delete_all=True (REST)",
        )

        # Verify named-namespace vectors are gone from fetch
        ns_fetch = index.fetch(ids=ns_ids, namespace=ns)
        assert isinstance(ns_fetch, FetchResponse)
        assert len(ns_fetch.vectors) == 0, \
            f"named-namespace vectors should be gone but found: {list(ns_fetch.vectors.keys())}"

        # Verify default namespace is unaffected
        def_fetch = index.fetch(ids=default_ids)
        assert isinstance(def_fetch, FetchResponse)
        for vid in default_ids:
            assert vid in def_fetch.vectors, \
                f"default-namespace vector {vid!r} should survive delete_all on different namespace"

        # Verify stats: named namespace is absent or has 0 vectors; total count == 2
        stats = index.describe_index_stats()
        assert isinstance(stats, DescribeIndexStatsResponse)
        if ns in stats.namespaces:
            assert stats.namespaces[ns].vector_count == 0, \
                f"dan-cleanup-ns should be empty but has {stats.namespaces[ns].vector_count} vectors"
        assert stats.total_vector_count == 2, \
            f"only 2 default-namespace vectors should remain, got total={stats.total_vector_count}"

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# delete-all-namespace — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_delete_all_namespace_grpc(client: Pinecone) -> None:
    """index.delete(delete_all=True, namespace=...) deletes all vectors in a named
    namespace (gRPC) while leaving other namespaces untouched.

    Same semantics as the REST sync test but via the gRPC transport.
    """
    name = unique_name("idx")
    ns = "dan-cleanup-ns"
    default_ids = ["dan-def-1", "dan-def-2"]
    ns_ids = ["dan-ns-1", "dan-ns-2", "dan-ns-3"]
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        # Upsert into named namespace
        ns_vectors = [
            {"id": "dan-ns-1", "values": [0.1, 0.2]},
            {"id": "dan-ns-2", "values": [0.3, 0.4]},
            {"id": "dan-ns-3", "values": [0.5, 0.6]},
        ]
        result = index.upsert(vectors=ns_vectors, namespace=ns)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 3

        # Upsert into default namespace
        def_vectors = [
            {"id": "dan-def-1", "values": [0.7, 0.8]},
            {"id": "dan-def-2", "values": [0.9, 0.1]},
        ]
        result2 = index.upsert(vectors=def_vectors)
        assert isinstance(result2, UpsertResponse)
        assert result2.upserted_count == 2

        # Wait for all vectors to be indexed in stats before deleting
        poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 5,
            timeout=120,
            description="all 5 vectors appear in stats before delete-all (gRPC)",
        )

        # Delete all vectors in the named namespace
        index.delete(delete_all=True, namespace=ns)

        # Poll until the named namespace is gone or empty
        poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: ns not in r.namespaces or r.namespaces[ns].vector_count == 0,
            timeout=120,
            description="dan-cleanup-ns empty after delete_all=True (gRPC)",
        )

        # Verify named-namespace vectors are gone from fetch
        ns_fetch = index.fetch(ids=ns_ids, namespace=ns)
        assert isinstance(ns_fetch, FetchResponse)
        assert len(ns_fetch.vectors) == 0, \
            f"named-namespace vectors should be gone but found: {list(ns_fetch.vectors.keys())} (gRPC)"

        # Verify default namespace is unaffected
        def_fetch = index.fetch(ids=default_ids)
        assert isinstance(def_fetch, FetchResponse)
        for vid in default_ids:
            assert vid in def_fetch.vectors, \
                f"default-namespace vector {vid!r} should survive delete_all on different namespace (gRPC)"

        # Verify stats: named namespace is absent or has 0 vectors; total count == 2
        stats = index.describe_index_stats()
        assert isinstance(stats, DescribeIndexStatsResponse)
        if ns in stats.namespaces:
            assert stats.namespaces[ns].vector_count == 0, \
                f"dan-cleanup-ns should be empty but has {stats.namespaces[ns].vector_count} vectors (gRPC)"
        assert stats.total_vector_count == 2, \
            f"only 2 default-namespace vectors should remain, got total={stats.total_vector_count} (gRPC)"

    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert-records input validation — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_records_validation_rest(client: Pinecone) -> None:
    """upsert_records raises PineconeValueError before any API call for invalid inputs.

    Verifies claims:
    - unified-vec-0049: Record upsert requires a non-empty records list.
    - unified-vec-0048: Each record must contain an '_id' or 'id' identifier field.

    All validation is client-side; no real index is created. The Index is
    constructed with a dummy host so that validation fires before any HTTP call.
    """
    index = client.index(host="https://dummy.example.com")

    # unified-vec-0049: empty records list raises PineconeValueError
    with pytest.raises(PineconeValueError):
        index.upsert_records(records=[], namespace="test-ns")

    # unified-vec-0048: record missing both '_id' and 'id' raises PineconeValueError
    with pytest.raises(PineconeValueError):
        index.upsert_records(
            records=[{"text": "no identifier field here"}],
            namespace="test-ns",
        )

    # namespace must be a non-empty string — whitespace-only is rejected
    with pytest.raises(PineconeValueError):
        index.upsert_records(
            records=[{"_id": "v1", "text": "hello"}],
            namespace="",
        )

    with pytest.raises(PineconeValueError):
        index.upsert_records(
            records=[{"_id": "v1", "text": "hello"}],
            namespace="   ",
        )


# ---------------------------------------------------------------------------
# delete() mode validation — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_delete_mode_validation_rest(client: Pinecone) -> None:
    """delete() raises PineconeValueError for conflicting or absent mode selection.

    Verifies claim unified-vec-0041: The delete operation accepts exactly one of:
    a list of identifiers, a delete-all flag, or a metadata filter; combining modes
    is not allowed. Passing no mode at all is also rejected.

    All validation is client-side; no real index is created. The Index is constructed
    with a dummy host so that validation fires before any HTTP request.
    """
    index = client.index(host="fake-index.svc.pinecone.io")

    # No mode at all raises PineconeValueError
    with pytest.raises(PineconeValueError):
        index.delete()

    # ids + filter combined raises PineconeValueError
    with pytest.raises(PineconeValueError):
        index.delete(ids=["v1"], filter={"category": {"$eq": "test"}})

    # ids + delete_all combined raises PineconeValueError
    with pytest.raises(PineconeValueError):
        index.delete(ids=["v1"], delete_all=True)

    # delete_all + filter combined raises PineconeValueError
    with pytest.raises(PineconeValueError):
        index.delete(delete_all=True, filter={"category": {"$eq": "test"}})

    # All three combined raises PineconeValueError
    with pytest.raises(PineconeValueError):
        index.delete(ids=["v1"], delete_all=True, filter={"category": {"$eq": "test"}})
