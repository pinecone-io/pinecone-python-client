"""Integration tests for extended data-plane operations (sync REST + gRPC).

Phase 2 Tier 1: metadata-filter, sparse-vectors, query-by-id, fetch-missing-ids,
include-values-metadata, query-namespaces.
"""
# area tags covered: metadata-filter, sparse-vectors, query-by-id, fetch-missing-ids,
# include-values-metadata, query-namespaces

from __future__ import annotations

import pytest

from pinecone import Field, Pinecone
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.responses import (
    FetchResponse,
    QueryResponse,
    UpsertResponse,
)
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import ScoredVector, Vector
from tests.integration.conftest import cleanup_resource, poll_until, unique_name

# ---------------------------------------------------------------------------
# metadata-filter — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_metadata_filter_rest(client: Pinecone) -> None:
    """Query with metadata filters ($eq, $in) returns only matching vectors (REST sync)."""
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

        # Upsert vectors with metadata
        index.upsert(
            vectors=[
                {"id": "mf-v1", "values": [0.1, 0.2], "metadata": {"genre": "comedy", "year": 2020}},
                {"id": "mf-v2", "values": [0.3, 0.4], "metadata": {"genre": "action", "year": 2021}},
                {"id": "mf-v3", "values": [0.5, 0.6], "metadata": {"genre": "comedy", "year": 2022}},
            ]
        )

        # Wait for all 3 vectors to be queryable (eventual consistency)
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10),
            check_fn=lambda r: len(r.matches) >= 3,
            timeout=120,
            description="all 3 vectors queryable before filter tests",
        )

        # Test $eq filter: only comedy vectors should return
        comedy_result = index.query(
            vector=[0.1, 0.2],
            top_k=10,
            filter={"genre": {"$eq": "comedy"}},
            include_metadata=True,
        )
        assert isinstance(comedy_result, QueryResponse)
        comedy_ids = {m.id for m in comedy_result.matches}
        assert "mf-v1" in comedy_ids
        assert "mf-v3" in comedy_ids
        assert "mf-v2" not in comedy_ids

        # Test $in filter: only action vector should return
        action_result = index.query(
            vector=[0.1, 0.2],
            top_k=10,
            filter={"genre": {"$in": ["action", "thriller"]}},
            include_metadata=True,
        )
        assert isinstance(action_result, QueryResponse)
        action_ids = {m.id for m in action_result.matches}
        assert "mf-v2" in action_ids
        assert "mf-v1" not in action_ids
        assert "mf-v3" not in action_ids

        # Verify metadata is present in results (since include_metadata=True)
        for match in comedy_result.matches:
            assert isinstance(match, ScoredVector)
            assert match.metadata is not None
            assert match.metadata.get("genre") == "comedy"
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# metadata-filter — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_metadata_filter_grpc(client: Pinecone) -> None:
    """Query with metadata filter via GrpcIndex returns only matching vectors."""
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
                {"id": "mf-v1", "values": [0.1, 0.2], "metadata": {"genre": "comedy"}},
                {"id": "mf-v2", "values": [0.3, 0.4], "metadata": {"genre": "action"}},
            ]
        )

        result = index.query(
            vector=[0.1, 0.2],
            top_k=10,
            filter={"genre": {"$eq": "comedy"}},
        )
        assert isinstance(result, QueryResponse)
        ids = {m.id for m in result.matches}
        assert "mf-v1" in ids
        assert "mf-v2" not in ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# sparse-vectors — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_sparse_vectors_rest(client: Pinecone) -> None:
    """Upsert hybrid (dense+sparse) vectors; fetch returns sparse_values; query with sparse_vector works (REST sync)."""
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

        # Upsert vectors with both dense values and sparse values
        upsert_resp = index.upsert(
            vectors=[
                {
                    "id": "sv-v1",
                    "values": [0.1, 0.2, 0.3, 0.4],
                    "sparse_values": {"indices": [0, 5], "values": [0.5, 0.8]},
                },
                {
                    "id": "sv-v2",
                    "values": [0.5, 0.6, 0.7, 0.8],
                    "sparse_values": {"indices": [2, 7], "values": [0.3, 0.9]},
                },
            ]
        )
        assert isinstance(upsert_resp, UpsertResponse)
        assert upsert_resp.upserted_count == 2

        # Wait for vectors to be fetchable (eventual consistency)
        poll_until(
            query_fn=lambda: index.fetch(ids=["sv-v1", "sv-v2"]),
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="sparse vectors fetchable",
        )

        # Fetch and verify sparse_values are returned
        fetch_resp = index.fetch(ids=["sv-v1", "sv-v2"])
        assert isinstance(fetch_resp, FetchResponse)
        assert "sv-v1" in fetch_resp.vectors
        v1 = fetch_resp.vectors["sv-v1"]
        assert isinstance(v1, Vector)
        assert v1.sparse_values is not None
        assert isinstance(v1.sparse_values, SparseValues)
        assert v1.sparse_values.indices == [0, 5]
        assert len(v1.sparse_values.values) == 2
        assert abs(v1.sparse_values.values[0] - 0.5) < 1e-4
        assert abs(v1.sparse_values.values[1] - 0.8) < 1e-4

        # Also verify v2 has sparse values
        v2 = fetch_resp.vectors["sv-v2"]
        assert v2.sparse_values is not None
        assert isinstance(v2.sparse_values, SparseValues)
        assert v2.sparse_values.indices == [2, 7]

        # Query with a sparse vector and verify matches are returned
        query_resp = index.query(
            vector=[0.1, 0.2, 0.3, 0.4],
            sparse_vector={"indices": [0, 5], "values": [0.5, 0.8]},
            top_k=5,
            include_values=True,
        )
        assert isinstance(query_resp, QueryResponse)
        assert len(query_resp.matches) >= 1
        match_ids = {m.id for m in query_resp.matches}
        # sv-v1 shares the same sparse indices — it should rank highly
        assert "sv-v1" in match_ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query-by-id — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_query_by_id_rest(client: Pinecone) -> None:
    """Query by stored vector ID returns a QueryResponse with the same structure as query-by-vector (REST sync)."""
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

        # Upsert 3 vectors
        index.upsert(
            vectors=[
                {"id": "qbi-v1", "values": [0.1, 0.2]},
                {"id": "qbi-v2", "values": [0.3, 0.4]},
                {"id": "qbi-v3", "values": [0.9, 0.1]},
            ]
        )

        # Wait for all 3 vectors to be queryable before querying by ID
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10),
            check_fn=lambda r: len(r.matches) >= 3,
            timeout=120,
            description="all 3 vectors queryable before query-by-id",
        )

        # Query by ID — use qbi-v1 as the query seed
        result = index.query(id="qbi-v1", top_k=3)
        assert isinstance(result, QueryResponse)
        assert len(result.matches) >= 1

        # Each match must have id and score
        for match in result.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)

        # The top match should be the query vector itself
        match_ids = [m.id for m in result.matches]
        assert "qbi-v1" in match_ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query-by-id — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_query_by_id_grpc(client: Pinecone) -> None:
    """Query by stored vector ID returns a QueryResponse via GrpcIndex."""
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

        # Upsert 3 vectors
        index.upsert(
            vectors=[
                {"id": "qbi-v1", "values": [0.1, 0.2]},
                {"id": "qbi-v2", "values": [0.3, 0.4]},
                {"id": "qbi-v3", "values": [0.9, 0.1]},
            ]
        )

        # Wait until vectors are queryable
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10),
            check_fn=lambda r: len(r.matches) >= 3,
            timeout=120,
            description="all 3 vectors queryable (grpc) before query-by-id",
        )

        # Query by ID
        result = index.query(id="qbi-v1", top_k=3)
        assert isinstance(result, QueryResponse)
        assert len(result.matches) >= 1

        for match in result.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)

        match_ids = [m.id for m in result.matches]
        assert "qbi-v1" in match_ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# fetch-missing-ids — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_fetch_missing_ids_rest(client: Pinecone) -> None:
    """fetch() with a mix of existing and non-existent IDs returns only existing vectors, no error (REST sync)."""
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

        # Upsert 2 known vectors
        index.upsert(
            vectors=[
                {"id": "fmi-v1", "values": [0.1, 0.2]},
                {"id": "fmi-v2", "values": [0.3, 0.4]},
            ]
        )

        # Wait for both vectors to be fetchable (eventual consistency)
        poll_until(
            query_fn=lambda: index.fetch(ids=["fmi-v1", "fmi-v2"]),
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="both vectors fetchable before missing-id test",
        )

        # Fetch with a mix of existing and non-existent IDs — no error expected
        fetch_resp = index.fetch(ids=["fmi-v1", "fmi-v2", "fmi-does-not-exist"])
        assert isinstance(fetch_resp, FetchResponse)

        # Only existing vectors are returned
        assert "fmi-v1" in fetch_resp.vectors
        assert "fmi-v2" in fetch_resp.vectors
        assert "fmi-does-not-exist" not in fetch_resp.vectors
        assert len(fetch_resp.vectors) == 2

        # Each returned vector has correct structure
        v1 = fetch_resp.vectors["fmi-v1"]
        assert isinstance(v1, Vector)
        assert v1.id == "fmi-v1"
        assert len(v1.values) == 2
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# fetch-missing-ids — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_fetch_missing_ids_grpc(client: Pinecone) -> None:
    """fetch() with a mix of existing and non-existent IDs returns only existing vectors, no error (gRPC)."""
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

        # Upsert 2 known vectors
        index.upsert(
            vectors=[
                {"id": "fmi-v1", "values": [0.1, 0.2]},
                {"id": "fmi-v2", "values": [0.3, 0.4]},
            ]
        )

        # Wait for both vectors to be fetchable
        poll_until(
            query_fn=lambda: index.fetch(ids=["fmi-v1", "fmi-v2"]),
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="both vectors fetchable (grpc) before missing-id test",
        )

        # Fetch with a mix of existing and non-existent IDs — no error expected
        fetch_resp = index.fetch(ids=["fmi-v1", "fmi-v2", "fmi-does-not-exist"])
        assert isinstance(fetch_resp, FetchResponse)

        # Only existing vectors are returned
        assert "fmi-v1" in fetch_resp.vectors
        assert "fmi-v2" in fetch_resp.vectors
        assert "fmi-does-not-exist" not in fetch_resp.vectors
        assert len(fetch_resp.vectors) == 2
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# sparse-vectors — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_sparse_vectors_grpc(client: Pinecone) -> None:
    """Upsert hybrid (dense+sparse) vectors; fetch returns sparse_values; query with sparse_vector works (gRPC)."""
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

        # Upsert vectors with both dense values and sparse values
        upsert_resp = index.upsert(
            vectors=[
                {
                    "id": "sv-v1",
                    "values": [0.1, 0.2, 0.3, 0.4],
                    "sparse_values": {"indices": [0, 5], "values": [0.5, 0.8]},
                },
                {
                    "id": "sv-v2",
                    "values": [0.5, 0.6, 0.7, 0.8],
                    "sparse_values": {"indices": [2, 7], "values": [0.3, 0.9]},
                },
            ]
        )
        assert isinstance(upsert_resp, UpsertResponse)
        assert upsert_resp.upserted_count == 2

        # Wait for vectors to be fetchable (eventual consistency)
        poll_until(
            query_fn=lambda: index.fetch(ids=["sv-v1", "sv-v2"]),
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="sparse vectors fetchable via gRPC",
        )

        # Fetch and verify sparse_values are returned
        fetch_resp = index.fetch(ids=["sv-v1", "sv-v2"])
        assert isinstance(fetch_resp, FetchResponse)
        assert "sv-v1" in fetch_resp.vectors
        v1 = fetch_resp.vectors["sv-v1"]
        assert isinstance(v1, Vector)
        assert v1.sparse_values is not None
        assert isinstance(v1.sparse_values, SparseValues)
        assert v1.sparse_values.indices == [0, 5]

        # Query with a sparse vector and verify matches are returned
        query_resp = index.query(
            vector=[0.1, 0.2, 0.3, 0.4],
            sparse_vector={"indices": [0, 5], "values": [0.5, 0.8]},
            top_k=5,
        )
        assert isinstance(query_resp, QueryResponse)
        assert len(query_resp.matches) >= 1
        assert "sv-v1" in {m.id for m in query_resp.matches}
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# include-values-metadata — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_include_values_metadata_rest(client: Pinecone) -> None:
    """Query with include_values=True/include_metadata=True returns values and metadata on matches;
    query with defaults returns empty values and None metadata (REST sync)."""
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

        # Upsert vectors with metadata
        index.upsert(
            vectors=[
                {"id": "ivm-v1", "values": [0.1, 0.2, 0.3], "metadata": {"color": "red", "rank": 1}},
                {"id": "ivm-v2", "values": [0.4, 0.5, 0.6], "metadata": {"color": "blue", "rank": 2}},
            ]
        )

        # Wait for both vectors to be queryable (eventual consistency)
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2, 0.3], top_k=10),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="both vectors queryable before include-values-metadata test",
        )

        # Query with include_values=True, include_metadata=True
        result = index.query(
            vector=[0.1, 0.2, 0.3],
            top_k=5,
            include_values=True,
            include_metadata=True,
        )
        assert isinstance(result, QueryResponse)
        assert len(result.matches) >= 1

        top_match = result.matches[0]
        assert isinstance(top_match, ScoredVector)
        # values should be a non-empty list of floats
        assert isinstance(top_match.values, list)
        assert len(top_match.values) == 3
        assert all(isinstance(v, float) for v in top_match.values)
        # metadata should be a dict
        assert isinstance(top_match.metadata, dict)
        assert "color" in top_match.metadata
        assert "rank" in top_match.metadata

        # Verify all matches have values and metadata
        for match in result.matches:
            assert len(match.values) == 3
            assert match.metadata is not None

        # Query with defaults (no include_values, no include_metadata)
        default_result = index.query(vector=[0.1, 0.2, 0.3], top_k=5)
        assert isinstance(default_result, QueryResponse)
        assert len(default_result.matches) >= 1

        default_match = default_result.matches[0]
        assert isinstance(default_match, ScoredVector)
        # Default: values should be empty, metadata should be None
        assert default_match.values == []
        assert default_match.metadata is None
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# include-values-metadata — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_include_values_metadata_grpc(client: Pinecone) -> None:
    """Query with include_values=True/include_metadata=True returns values and metadata (gRPC)."""
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

        # Upsert vectors with metadata
        index.upsert(
            vectors=[
                {"id": "ivm-v1", "values": [0.1, 0.2, 0.3], "metadata": {"color": "red"}},
                {"id": "ivm-v2", "values": [0.4, 0.5, 0.6], "metadata": {"color": "blue"}},
            ]
        )

        # Wait for vectors to be queryable (eventual consistency)
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2, 0.3], top_k=10),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="both vectors queryable (grpc) before include-values-metadata test",
        )

        # Query with include_values=True, include_metadata=True
        result = index.query(
            vector=[0.1, 0.2, 0.3],
            top_k=5,
            include_values=True,
            include_metadata=True,
        )
        assert isinstance(result, QueryResponse)
        assert len(result.matches) >= 1

        top_match = result.matches[0]
        assert isinstance(top_match, ScoredVector)
        assert isinstance(top_match.values, list)
        assert len(top_match.values) == 3
        assert isinstance(top_match.metadata, dict)
        assert "color" in top_match.metadata

        # Query with defaults — values empty, metadata None
        default_result = index.query(vector=[0.1, 0.2, 0.3], top_k=5)
        assert isinstance(default_result, QueryResponse)
        assert len(default_result.matches) >= 1

        default_match = default_result.matches[0]
        assert default_match.values == []
        assert default_match.metadata is None
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query-namespaces — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_query_namespaces_rest(client: Pinecone) -> None:
    """query_namespaces() fans out queries across multiple namespaces and merges results (REST sync)."""
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

        # Upsert different vectors into two named namespaces
        index.upsert(
            vectors=[
                {"id": "qn-ns1-v1", "values": [0.1, 0.2]},
                {"id": "qn-ns1-v2", "values": [0.3, 0.4]},
            ],
            namespace="qn-ns1",
        )
        index.upsert(
            vectors=[
                {"id": "qn-ns2-v1", "values": [0.5, 0.6]},
                {"id": "qn-ns2-v2", "values": [0.7, 0.8]},
            ],
            namespace="qn-ns2",
        )

        # Wait until at least one vector from each namespace is queryable
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace="qn-ns1"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="ns1 vectors queryable before query_namespaces",
        )
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace="qn-ns2"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="ns2 vectors queryable before query_namespaces",
        )

        # Call query_namespaces across both namespaces
        results = index.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=["qn-ns1", "qn-ns2"],
            metric="cosine",
            top_k=5,
        )

        # Verify result type and structure
        assert isinstance(results, QueryNamespacesResults)

        # Merged matches list: sorted by score (descending for cosine), up to top_k
        assert isinstance(results.matches, list)
        assert len(results.matches) >= 1

        # Each match is a ScoredVector with id and score
        for match in results.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)

        # Matches come from both namespaces
        match_ids = {m.id for m in results.matches}
        assert len(match_ids & {"qn-ns1-v1", "qn-ns1-v2"}) >= 1
        assert len(match_ids & {"qn-ns2-v1", "qn-ns2-v2"}) >= 1

        # Scores should be in descending order (cosine: higher is better)
        scores = [m.score for m in results.matches]
        assert scores == sorted(scores, reverse=True)

        # usage has read_units (sum across namespaces)
        assert results.usage is not None
        assert isinstance(results.usage.read_units, int)
        assert results.usage.read_units >= 2  # at least 1 per namespace

        # ns_usage has per-namespace usage
        assert isinstance(results.ns_usage, dict)
        assert "qn-ns1" in results.ns_usage
        assert "qn-ns2" in results.ns_usage
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# metadata-filter numeric comparisons ($gt, $gte, $lt, $lte, $ne) — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_metadata_filter_numeric_operators_rest(client: Pinecone) -> None:
    """Numeric comparison operators ($gt, $gte, $lt, $lte, $ne) filter vectors correctly (REST sync).

    Verifies unified-filter-0001: Can build metadata filters using not-equal,
    greater-than, greater-than-or-equal, less-than, and less-than-or-equal operators.
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

        # Upsert 4 vectors with distinct integer year metadata values
        index.upsert(
            vectors=[
                {"id": "nf-v1", "values": [0.1, 0.2], "metadata": {"year": 2019}},
                {"id": "nf-v2", "values": [0.3, 0.4], "metadata": {"year": 2020}},
                {"id": "nf-v3", "values": [0.5, 0.6], "metadata": {"year": 2021}},
                {"id": "nf-v4", "values": [0.7, 0.8], "metadata": {"year": 2022}},
            ]
        )

        # Wait until all 4 vectors are visible
        poll_until(
            query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10),
            check_fn=lambda r: len(r.matches) >= 4,
            timeout=120,
            description="all 4 vectors queryable before numeric filter tests",
        )

        # $gt: 2020 — only years strictly greater than 2020 (2021, 2022)
        gt_result = index.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter={"year": {"$gt": 2020}},
            include_metadata=True,
        )
        assert isinstance(gt_result, QueryResponse)
        gt_ids = {m.id for m in gt_result.matches}
        assert "nf-v3" in gt_ids
        assert "nf-v4" in gt_ids
        assert "nf-v1" not in gt_ids
        assert "nf-v2" not in gt_ids

        # $gte: 2020 — years >= 2020 (2020, 2021, 2022)
        gte_result = index.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter={"year": {"$gte": 2020}},
            include_metadata=True,
        )
        assert isinstance(gte_result, QueryResponse)
        gte_ids = {m.id for m in gte_result.matches}
        assert "nf-v2" in gte_ids
        assert "nf-v3" in gte_ids
        assert "nf-v4" in gte_ids
        assert "nf-v1" not in gte_ids

        # $lt: 2021 — years strictly less than 2021 (2019, 2020)
        lt_result = index.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter={"year": {"$lt": 2021}},
            include_metadata=True,
        )
        assert isinstance(lt_result, QueryResponse)
        lt_ids = {m.id for m in lt_result.matches}
        assert "nf-v1" in lt_ids
        assert "nf-v2" in lt_ids
        assert "nf-v3" not in lt_ids
        assert "nf-v4" not in lt_ids

        # $lte: 2021 — years <= 2021 (2019, 2020, 2021)
        lte_result = index.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter={"year": {"$lte": 2021}},
            include_metadata=True,
        )
        assert isinstance(lte_result, QueryResponse)
        lte_ids = {m.id for m in lte_result.matches}
        assert "nf-v1" in lte_ids
        assert "nf-v2" in lte_ids
        assert "nf-v3" in lte_ids
        assert "nf-v4" not in lte_ids

        # $ne: 2021 — years not equal to 2021 (2019, 2020, 2022)
        ne_result = index.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter={"year": {"$ne": 2021}},
            include_metadata=True,
        )
        assert isinstance(ne_result, QueryResponse)
        ne_ids = {m.id for m in ne_result.matches}
        assert "nf-v1" in ne_ids
        assert "nf-v2" in ne_ids
        assert "nf-v4" in ne_ids
        assert "nf-v3" not in ne_ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# metadata-filter logical operators ($nin, &, |) via Field builder — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_metadata_filter_logical_operators_rest(client: Pinecone) -> None:
    """$nin, logical AND (&), and logical OR (|) via the Field builder filter correctly (REST sync).

    Verifies:
      unified-filter-0002 — not-in-list ($nin) operator
      unified-filter-0004 — combining filters with & (AND) and | (OR)
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

        # Upsert 4 vectors with genre + year metadata
        index.upsert(
            vectors=[
                {"id": "lo-v1", "values": [0.1, 0.2], "metadata": {"genre": "comedy", "year": 2020}},
                {"id": "lo-v2", "values": [0.3, 0.4], "metadata": {"genre": "action", "year": 2021}},
                {"id": "lo-v3", "values": [0.5, 0.6], "metadata": {"genre": "comedy", "year": 2022}},
                {"id": "lo-v4", "values": [0.7, 0.8], "metadata": {"genre": "horror", "year": 2021}},
            ]
        )

        # Wait until all 4 vectors are visible
        poll_until(
            query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10),
            check_fn=lambda r: len(r.matches) >= 4,
            timeout=120,
            description="all 4 vectors queryable before logical filter tests",
        )

        # $nin: genres NOT in ["horror", "action"] → should return only comedy vectors (v1, v3)
        nin_filter = Field("genre").not_in(["horror", "action"])
        nin_result = index.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter=nin_filter.to_dict(),
            include_metadata=True,
        )
        assert isinstance(nin_result, QueryResponse)
        nin_ids = {m.id for m in nin_result.matches}
        assert "lo-v1" in nin_ids
        assert "lo-v3" in nin_ids
        assert "lo-v2" not in nin_ids
        assert "lo-v4" not in nin_ids

        # & (AND): genre == comedy AND year >= 2021 → should return only v3 (2022)
        and_filter = (Field("genre") == "comedy") & Field("year").gte(2021)
        and_result = index.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter=and_filter.to_dict(),
            include_metadata=True,
        )
        assert isinstance(and_result, QueryResponse)
        and_ids = {m.id for m in and_result.matches}
        assert "lo-v3" in and_ids
        assert "lo-v1" not in and_ids  # comedy but year 2020 < 2021
        assert "lo-v2" not in and_ids  # action
        assert "lo-v4" not in and_ids  # horror

        # | (OR): genre == horror OR genre == action → should return v2 and v4
        or_filter = (Field("genre") == "horror") | (Field("genre") == "action")
        or_result = index.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter=or_filter.to_dict(),
            include_metadata=True,
        )
        assert isinstance(or_result, QueryResponse)
        or_ids = {m.id for m in or_result.matches}
        assert "lo-v2" in or_ids
        assert "lo-v4" in or_ids
        assert "lo-v1" not in or_ids
        assert "lo-v3" not in or_ids

        # Verify to_dict() produces the correct wire-format dicts
        assert and_filter.to_dict() == {
            "$and": [{"genre": {"$eq": "comedy"}}, {"year": {"$gte": 2021}}]
        }
        assert or_filter.to_dict() == {
            "$or": [{"genre": {"$eq": "horror"}}, {"genre": {"$eq": "action"}}]
        }
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# metadata-filter logical operators ($nin, &, |) via Field builder — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_metadata_filter_logical_operators_grpc(client: Pinecone) -> None:
    """$nin and logical AND (&) via the Field builder filter correctly (gRPC).

    Verifies unified-filter-0002 and unified-filter-0004 on the gRPC transport.
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

        index.upsert(
            vectors=[
                {"id": "lo-v1", "values": [0.1, 0.2], "metadata": {"genre": "comedy", "year": 2020}},
                {"id": "lo-v2", "values": [0.3, 0.4], "metadata": {"genre": "action", "year": 2021}},
                {"id": "lo-v3", "values": [0.5, 0.6], "metadata": {"genre": "comedy", "year": 2022}},
                {"id": "lo-v4", "values": [0.7, 0.8], "metadata": {"genre": "horror", "year": 2021}},
            ]
        )

        poll_until(
            query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10),
            check_fn=lambda r: len(r.matches) >= 4,
            timeout=120,
            description="all 4 vectors queryable (grpc) before logical filter tests",
        )

        # $nin: genres NOT in ["horror", "action"] → comedy vectors only (v1, v3)
        nin_filter = Field("genre").not_in(["horror", "action"])
        nin_result = index.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter=nin_filter.to_dict(),
        )
        assert isinstance(nin_result, QueryResponse)
        nin_ids = {m.id for m in nin_result.matches}
        assert "lo-v1" in nin_ids
        assert "lo-v3" in nin_ids
        assert "lo-v2" not in nin_ids
        assert "lo-v4" not in nin_ids

        # & (AND): genre == comedy AND year >= 2021 → v3 only
        and_filter = (Field("genre") == "comedy") & Field("year").gte(2021)
        and_result = index.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter=and_filter.to_dict(),
        )
        assert isinstance(and_result, QueryResponse)
        and_ids = {m.id for m in and_result.matches}
        assert "lo-v3" in and_ids
        assert "lo-v1" not in and_ids
        assert "lo-v2" not in and_ids
        assert "lo-v4" not in and_ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")
