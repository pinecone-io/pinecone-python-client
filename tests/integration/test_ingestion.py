"""Integration tests for deep data-ingestion scenarios (sync REST + gRPC).

Phase 3 area tags: upsert-formats, upsert-batch, upsert-overwrite,
upsert-records, upsert-records-batch, update-metadata, update-sparse,
update-by-filter, delete-by-filter, delete-all-namespace
"""

from __future__ import annotations

import math
import pytest

from pinecone import Pinecone, Vector
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.responses import FetchResponse, UpsertResponse
from tests.integration.conftest import cleanup_resource, poll_until, unique_name


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
