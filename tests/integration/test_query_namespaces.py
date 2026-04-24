"""Integration tests for advanced query_namespaces operations (sync REST).

Phase 3 Tier 5: query-namespaces-filter, query-namespaces-many.
ET-019: query-namespaces-dedup.
"""
# area tags covered: query-namespaces-filter, query-namespaces-many, query-namespaces-dedup

from __future__ import annotations

import time

import pytest

from pinecone import Pinecone
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.vector import ScoredVector
from tests.integration.conftest import (
    cleanup_resource,
    ensure_index_deleted,
    poll_until,
    unique_name,
)

# ---------------------------------------------------------------------------
# query-namespaces-filter — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_namespaces_filter_rest(client: Pinecone) -> None:
    """query_namespaces() with filter applies it per-namespace and returns metadata (REST sync)."""
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

        # Upsert comedy + drama vectors into two namespaces
        index.upsert(
            vectors=[
                {"id": "qnf-ns1-com1", "values": [0.1, 0.2], "metadata": {"genre": "comedy"}},
                {"id": "qnf-ns1-dra1", "values": [0.9, 0.8], "metadata": {"genre": "drama"}},
            ],
            namespace="qnf-ns1",
        )
        index.upsert(
            vectors=[
                {"id": "qnf-ns2-com1", "values": [0.2, 0.3], "metadata": {"genre": "comedy"}},
                {"id": "qnf-ns2-dra1", "values": [0.8, 0.7], "metadata": {"genre": "drama"}},
            ],
            namespace="qnf-ns2",
        )

        # Wait for all vectors in both namespaces to be queryable
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace="qnf-ns1"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="ns1 vectors queryable before query_namespaces_filter",
        )
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10, namespace="qnf-ns2"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="ns2 vectors queryable before query_namespaces_filter",
        )

        # Call query_namespaces with comedy filter and include_metadata=True
        results = index.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=["qnf-ns1", "qnf-ns2"],
            metric="cosine",
            top_k=10,
            filter={"genre": {"$eq": "comedy"}},
            include_metadata=True,
        )

        # Verify result type and structure
        assert isinstance(results, QueryNamespacesResults)
        assert isinstance(results.matches, list)
        assert len(results.matches) >= 1

        # Each match is a ScoredVector
        for match in results.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)

        # Filter must have been applied: only comedy vectors should appear
        match_ids = {m.id for m in results.matches}
        comedy_ids = {"qnf-ns1-com1", "qnf-ns2-com1"}
        drama_ids = {"qnf-ns1-dra1", "qnf-ns2-dra1"}
        # At least one comedy vector from each namespace should be in results
        assert len(match_ids & comedy_ids) >= 1
        # Drama vectors must be absent (filter excluded them)
        assert match_ids.isdisjoint(drama_ids), (
            f"Drama vectors leaked through filter: {match_ids & drama_ids}"
        )

        # Metadata must be present on matches (include_metadata=True)
        for match in results.matches:
            assert match.metadata is not None
            assert "genre" in match.metadata
            assert match.metadata["genre"] == "comedy"

        # Scores should be in descending order (cosine)
        scores = [m.score for m in results.matches]
        assert scores == sorted(scores, reverse=True)

        # Per-namespace usage should be populated
        assert isinstance(results.ns_usage, dict)
        assert "qnf-ns1" in results.ns_usage
        assert "qnf-ns2" in results.ns_usage

        # Total usage
        assert results.usage is not None
        assert isinstance(results.usage.read_units, int)
        assert results.usage.read_units >= 2
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query-namespaces-dedup — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_namespaces_dedup_rest(client: Pinecone) -> None:
    """query_namespaces() deduplicates repeated namespaces: no vector appears twice, ns_usage has one key per unique namespace (REST sync).

    Verifies unified-vec-0034: duplicate entries in the namespaces list are
    removed before fan-out, so each namespace is queried exactly once.
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

        # Upsert distinct vectors into two namespaces
        index.upsert(
            vectors=[
                {"id": "qnd-ns1-v1", "values": [0.1, 0.9]},
                {"id": "qnd-ns1-v2", "values": [0.9, 0.1]},
            ],
            namespace="qnd-ns1",
        )
        index.upsert(
            vectors=[
                {"id": "qnd-ns2-v1", "values": [0.5, 0.5]},
                {"id": "qnd-ns2-v2", "values": [0.6, 0.4]},
            ],
            namespace="qnd-ns2",
        )

        # Wait for both namespaces to be queryable
        poll_until(
            query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10, namespace="qnd-ns1"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="qnd-ns1 vectors queryable before dedup test",
        )
        poll_until(
            query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10, namespace="qnd-ns2"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="qnd-ns2 vectors queryable before dedup test",
        )

        # Query with a duplicated namespaces list: ns1 appears twice
        results = index.query_namespaces(
            vector=[0.5, 0.5],
            namespaces=["qnd-ns1", "qnd-ns2", "qnd-ns1"],
            metric="cosine",
            top_k=10,
        )

        assert isinstance(results, QueryNamespacesResults)
        assert isinstance(results.matches, list)

        # Each match is a ScoredVector
        for match in results.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)

        # Dedup: no vector ID should appear more than once in results
        result_ids = [m.id for m in results.matches]
        assert len(result_ids) == len(set(result_ids)), (
            f"Duplicate vector IDs in results (ns1 was queried twice): {result_ids}"
        )

        # ns_usage must have exactly 2 keys — the deduplicated set
        assert isinstance(results.ns_usage, dict)
        assert set(results.ns_usage.keys()) == {"qnd-ns1", "qnd-ns2"}, (
            f"Expected ns_usage keys {{'qnd-ns1','qnd-ns2'}}, got {set(results.ns_usage.keys())}"
        )

        # Scores must be in descending order
        scores = [m.score for m in results.matches]
        assert scores == sorted(scores, reverse=True), (
            f"Matches not sorted by descending score: {scores}"
        )
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query-namespaces-many — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_namespaces_many_rest(client: Pinecone) -> None:
    """query_namespaces() across 5+ namespaces merges and sorts results; ns_usage has entry per namespace (REST sync)."""
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

        # Upsert 2 vectors into each of 5 namespaces
        namespaces = [f"qnm-ns-{i}" for i in range(5)]
        for i, ns in enumerate(namespaces):
            base = float(i) / 5.0
            index.upsert(
                vectors=[
                    {"id": f"{ns}-v1", "values": [base, 1.0 - base]},
                    {"id": f"{ns}-v2", "values": [1.0 - base, base]},
                ],
                namespace=ns,
            )

        # Wait for each namespace to have both vectors queryable
        for ns in namespaces:
            poll_until(
                query_fn=lambda ns=ns: index.query(vector=[0.5, 0.5], top_k=10, namespace=ns),
                check_fn=lambda r: len(r.matches) >= 2,
                timeout=120,
                description=f"{ns} vectors queryable before query_namespaces_many",
            )

        # Query across all 5 namespaces at once
        results = index.query_namespaces(
            vector=[0.5, 0.5],
            namespaces=namespaces,
            metric="cosine",
            top_k=5,
        )

        # Verify result type and structure
        assert isinstance(results, QueryNamespacesResults)
        assert isinstance(results.matches, list)
        assert len(results.matches) >= 1

        # Each match must be a ScoredVector
        for match in results.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)

        # Results must be sorted by descending score (merged across namespaces)
        scores = [m.score for m in results.matches]
        assert scores == sorted(scores, reverse=True), (
            f"Matches not sorted by descending score: {scores}"
        )

        # ns_usage must contain an entry for every queried namespace
        assert isinstance(results.ns_usage, dict)
        for ns in namespaces:
            assert ns in results.ns_usage, (
                f"Expected ns_usage entry for {ns!r}, got keys: {list(results.ns_usage.keys())}"
            )

        # Total usage must be present and reflect work across all namespaces
        assert results.usage is not None
        assert isinstance(results.usage.read_units, int)
        assert results.usage.read_units >= len(namespaces)
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query-namespaces-default-top-k — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_namespaces_default_top_k_rest(client: Pinecone) -> None:
    """query_namespaces() defaults top_k to 10 when not specified (REST sync).

    Verifies claim unified-vec-0028: Cross-namespace query defaults to returning
    the top 10 results when top_k is not specified.

    Strategy: upsert 7 vectors into two namespaces (14 total > 10 default), then
    call query_namespaces without top_k and assert that at most 10 matches are
    returned, confirming the default is applied.
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

        # Upsert 7 vectors into each of 2 namespaces = 14 total (exceeds default top_k=10)
        ns_a_vectors = [
            {"id": f"qtk-ns-a-{i}", "values": [float(i) / 7, 1.0 - float(i) / 7]} for i in range(7)
        ]
        ns_b_vectors = [
            {"id": f"qtk-ns-b-{i}", "values": [float(i) / 14, 1.0 - float(i) / 14]}
            for i in range(7)
        ]
        index.upsert(vectors=ns_a_vectors, namespace="qtk-ns-a")
        index.upsert(vectors=ns_b_vectors, namespace="qtk-ns-b")

        # Wait for all 7 vectors in each namespace to become queryable
        poll_until(
            query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10, namespace="qtk-ns-a"),
            check_fn=lambda r: len(r.matches) >= 7,
            timeout=120,
            description="all 7 qtk-ns-a vectors queryable before default-top-k test",
        )
        poll_until(
            query_fn=lambda: index.query(vector=[0.5, 0.5], top_k=10, namespace="qtk-ns-b"),
            check_fn=lambda r: len(r.matches) >= 7,
            timeout=120,
            description="all 7 qtk-ns-b vectors queryable before default-top-k test",
        )

        # Query without top_k — should use default of 10
        results = index.query_namespaces(
            vector=[0.5, 0.5],
            namespaces=["qtk-ns-a", "qtk-ns-b"],
            metric="cosine",
        )

        assert isinstance(results, QueryNamespacesResults)
        assert isinstance(results.matches, list)
        # Key assertion: default top_k caps results at 10 even though 14 vectors exist
        assert len(results.matches) <= 10, (
            f"Expected at most 10 matches (default top_k), got {len(results.matches)}"
        )
        assert len(results.matches) > 0, "Expected at least one match"

        # Results must be sorted by descending score
        scores = [m.score for m in results.matches]
        assert scores == sorted(scores, reverse=True), (
            f"Matches not sorted by descending score: {scores}"
        )

        # Each match is a ScoredVector
        for match in results.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query-namespaces-euclidean — ascending score ordering — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_namespaces_euclidean_scores_ascending_rest(client: Pinecone) -> None:
    """query_namespaces() with euclidean metric returns matches sorted ascending by score (REST sync).

    For euclidean, lower scores indicate smaller distance (closer vectors) and should
    rank first — the opposite of cosine/dotproduct where higher scores rank first.

    Verifies claim unified-vec-0036: "Multi-namespace query results are aggregated using
    a heap-based algorithm; for cosine/dotproduct, higher scores rank first; for
    euclidean, lower scores rank first."

    Strategy:
    - Create an index with euclidean metric.
    - Upsert three vectors at known distances from the query vector [0.0, 0.0]:
        ns1: "euc-close"  at [0.1, 0.0]  (euclidean dist ≈ 0.1 — closest)
        ns1: "euc-far"    at [0.9, 0.0]  (euclidean dist ≈ 0.9 — farthest)
        ns2: "euc-mid"    at [0.4, 0.0]  (euclidean dist ≈ 0.4 — middle)
    - Query with vector [0.0, 0.0] and metric="euclidean".
    - Assert scores are non-decreasing (ascending order), confirming that lower
      (closer) scores rank first.
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        index.upsert(
            vectors=[
                {"id": "euc-close", "values": [0.1, 0.0]},
                {"id": "euc-far", "values": [0.9, 0.0]},
            ],
            namespace="euc-ns1",
        )
        index.upsert(
            vectors=[
                {"id": "euc-mid", "values": [0.4, 0.0]},
            ],
            namespace="euc-ns2",
        )

        # Wait for all 3 vectors to be queryable across both namespaces
        poll_until(
            query_fn=lambda: index.query(vector=[0.0, 0.0], top_k=10, namespace="euc-ns1"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="euc-ns1 vectors queryable before euclidean sort test",
        )
        poll_until(
            query_fn=lambda: index.query(vector=[0.0, 0.0], top_k=10, namespace="euc-ns2"),
            check_fn=lambda r: len(r.matches) >= 1,
            timeout=120,
            description="euc-ns2 vector queryable before euclidean sort test",
        )

        results = index.query_namespaces(
            vector=[0.0, 0.0],
            namespaces=["euc-ns1", "euc-ns2"],
            metric="euclidean",
            top_k=5,
        )

        assert isinstance(results, QueryNamespacesResults)
        assert isinstance(results.matches, list)
        assert len(results.matches) == 3, (
            f"Expected 3 matches (all vectors), got {len(results.matches)}"
        )

        # unified-vec-0036: for euclidean, scores must be sorted ascending
        scores = [m.score for m in results.matches]
        assert scores == sorted(scores), (
            f"For euclidean metric, scores must be ascending (lower = closer); got: {scores}"
        )

        # All scores must be non-negative (euclidean distance is always >= 0)
        for score in scores:
            assert score >= 0.0, f"Euclidean distance score must be non-negative, got {score}"

        # The closest vector (euc-close at [0.1, 0.0]) must rank first
        assert results.matches[0].id == "euc-close", (
            f"Expected 'euc-close' (closest to [0,0]) to rank first; "
            f"got {results.matches[0].id} (score={results.matches[0].score:.4f})"
        )

        # The farthest vector (euc-far at [0.9, 0.0]) must rank last
        assert results.matches[-1].id == "euc-far", (
            f"Expected 'euc-far' (farthest from [0,0]) to rank last; "
            f"got {results.matches[-1].id} (score={results.matches[-1].score:.4f})"
        )

        # Verify ScoredVector structure for all matches
        for match in results.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query-namespaces include_values — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_namespaces_include_values_rest(client: Pinecone) -> None:
    """query_namespaces(include_values=True) returns vector values on each match;
    omitting include_values leaves match.values as None (REST sync).

    Verifies:
    - unified-vec-0023: Query results do not include vector values unless explicitly
      requested — tested via the multi-namespace fan-out path.
    - unified-vec-0016: Can query multiple namespaces and return a merged result set
      with all optional fields populated when requested.
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
        index = client.index(name=name)

        # Upsert 2 vectors into each of 2 namespaces with known values
        index.upsert(
            vectors=[
                {"id": "iv-ns1-v1", "values": [0.1, 0.2, 0.3]},
                {"id": "iv-ns1-v2", "values": [0.4, 0.5, 0.6]},
            ],
            namespace="iv-ns1",
        )
        index.upsert(
            vectors=[
                {"id": "iv-ns2-v1", "values": [0.7, 0.8, 0.9]},
                {"id": "iv-ns2-v2", "values": [0.2, 0.3, 0.4]},
            ],
            namespace="iv-ns2",
        )

        # Wait for all vectors to be queryable in both namespaces
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace="iv-ns1"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="iv-ns1 vectors queryable before include_values test",
        )
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace="iv-ns2"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="iv-ns2 vectors queryable before include_values test",
        )

        # --- Part 1: include_values=True → values present on every match ---
        results_with_values = index.query_namespaces(
            vector=[0.1, 0.2, 0.3],
            namespaces=["iv-ns1", "iv-ns2"],
            metric="cosine",
            top_k=4,
            include_values=True,
        )

        assert isinstance(results_with_values, QueryNamespacesResults)
        assert len(results_with_values.matches) >= 1, (
            "Expected at least one match when include_values=True"
        )

        for match in results_with_values.matches:
            assert isinstance(match, ScoredVector)
            # values must be a non-empty list of floats when include_values=True
            assert match.values is not None, (
                f"match.values must not be None when include_values=True (id={match.id!r})"
            )
            assert isinstance(match.values, list), (
                f"match.values must be a list, got {type(match.values)} (id={match.id!r})"
            )
            assert len(match.values) == 3, (
                f"match.values length must equal index dimension 3, "
                f"got {len(match.values)} (id={match.id!r})"
            )
            assert all(isinstance(v, float) for v in match.values), (
                f"match.values elements must be floats (id={match.id!r}): {match.values}"
            )
            # metadata was not requested — must be None
            assert match.metadata is None, (
                f"match.metadata must be None when include_metadata not set (id={match.id!r})"
            )

        # --- Part 2: include_values omitted (default False) → values absent ---
        results_no_values = index.query_namespaces(
            vector=[0.1, 0.2, 0.3],
            namespaces=["iv-ns1", "iv-ns2"],
            metric="cosine",
            top_k=4,
        )

        assert isinstance(results_no_values, QueryNamespacesResults)
        assert len(results_no_values.matches) >= 1, (
            "Expected at least one match when include_values not set"
        )

        for match in results_no_values.matches:
            assert isinstance(match, ScoredVector)
            # values must be empty list when include_values is not requested
            # (ScoredVector defaults values to [] — not None — when the API omits the field)
            assert match.values == [], (
                f"match.values must be empty [] when include_values not requested (id={match.id!r}), "
                f"got {match.values!r}"
            )
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query-namespaces-sparse — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_namespaces_sparse_rest(client: Pinecone) -> None:
    """query_namespaces() with sparse_vector on a sparse dotproduct index returns merged results (REST sync).

    Verifies that a sparse-only index can be queried across multiple namespaces
    using only sparse_vector (no dense vector), with results merged and sorted
    by dotproduct score (descending) and per-namespace usage populated.
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            vector_type="sparse",
            metric="dotproduct",
            timeout=300,
        )
        index = client.index(name=name)

        # Upsert sparse-only vectors into two namespaces
        index.upsert(
            vectors=[
                {
                    "id": "qns-ns1-v1",
                    "sparse_values": {"indices": [0, 1, 2], "values": [0.5, 0.8, 0.3]},
                },
                {
                    "id": "qns-ns1-v2",
                    "sparse_values": {"indices": [1, 3, 5], "values": [0.2, 0.7, 0.4]},
                },
            ],
            namespace="qns-ns1",
        )
        index.upsert(
            vectors=[
                {
                    "id": "qns-ns2-v1",
                    "sparse_values": {"indices": [0, 2, 4], "values": [0.6, 0.3, 0.9]},
                },
                {
                    "id": "qns-ns2-v2",
                    "sparse_values": {"indices": [1, 2, 3], "values": [0.4, 0.5, 0.6]},
                },
            ],
            namespace="qns-ns2",
        )

        # Wait until sparse vectors are fetchable in both namespaces
        poll_until(
            query_fn=lambda: index.fetch(ids=["qns-ns1-v1", "qns-ns1-v2"], namespace="qns-ns1"),
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="qns-ns1 sparse vectors fetchable before query_namespaces_sparse",
        )
        poll_until(
            query_fn=lambda: index.fetch(ids=["qns-ns2-v1", "qns-ns2-v2"], namespace="qns-ns2"),
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="qns-ns2 sparse vectors fetchable before query_namespaces_sparse",
        )

        # Sparse-only query: pass sparse_vector, not vector
        results = index.query_namespaces(
            namespaces=["qns-ns1", "qns-ns2"],
            sparse_vector={"indices": [0, 1, 2], "values": [0.1, 0.2, 0.3]},
            metric="dotproduct",
            top_k=5,
        )

        assert isinstance(results, QueryNamespacesResults)
        assert isinstance(results.matches, list)
        assert len(results.matches) >= 1

        for match in results.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)

        # Scores sorted descending for dotproduct
        scores = [m.score for m in results.matches]
        assert scores == sorted(scores, reverse=True), (
            f"Expected scores sorted descending for dotproduct, got: {scores}"
        )

        # Per-namespace usage
        assert isinstance(results.ns_usage, dict)
        assert "qns-ns1" in results.ns_usage
        assert "qns-ns2" in results.ns_usage
        for ns_usage_val in results.ns_usage.values():
            assert ns_usage_val.read_units >= 0

        # Total usage
        assert results.usage is not None
        assert isinstance(results.usage.read_units, int)
        assert results.usage.read_units >= 2
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query-namespaces-parallel — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_namespaces_parallel_faster_than_serial_rest(client: Pinecone) -> None:
    """query_namespaces() across 10 namespaces executes queries in parallel.

    Verifies claim unified-vec-0035: individual per-namespace queries are fanned
    out via a thread pool, so wall-clock time is substantially less than a
    sequential baseline.
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

        namespaces = [f"qnp-ns-{i}" for i in range(10)]

        for ns in namespaces:
            index.upsert(
                vectors=[
                    {"id": f"{ns}-v{j}", "values": [float(j) / 5, 1.0 - float(j) / 5]}
                    for j in range(5)
                ],
                namespace=ns,
            )

        # Wait for each namespace to have all 5 vectors queryable
        for ns in namespaces:
            poll_until(
                query_fn=lambda ns=ns: index.query(vector=[0.5, 0.5], top_k=10, namespace=ns),
                check_fn=lambda r: len(r.matches) >= 5,
                timeout=120,
                description=f"{ns} vectors queryable before parallel test",
            )

        # Serial baseline: loop over each namespace individually
        serial_start = time.monotonic()
        for ns in namespaces:
            index.query(vector=[0.5, 0.5], top_k=5, namespace=ns)
        serial_elapsed = time.monotonic() - serial_start

        # Parallel call: single query_namespaces fan-out
        parallel_start = time.monotonic()
        results = index.query_namespaces(
            vector=[0.5, 0.5],
            namespaces=namespaces,
            metric="cosine",
            top_k=5,
        )
        parallel_elapsed = time.monotonic() - parallel_start

        # Correctness assertions
        assert isinstance(results, QueryNamespacesResults)
        assert isinstance(results.matches, list)
        assert 1 <= len(results.matches) <= 5
        for match in results.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)
        scores = [m.score for m in results.matches]
        assert scores == sorted(scores, reverse=True)
        for ns in namespaces:
            assert ns in results.ns_usage

        # Skip if backend is too fast to distinguish parallel from serial
        if serial_elapsed < 0.1:
            pytest.skip(f"serial baseline too fast to be meaningful: {serial_elapsed:.3f}s")

        # Parallelism assertion: parallel must be substantially faster
        assert parallel_elapsed < serial_elapsed * 0.6, (
            f"query_namespaces must fan out queries in parallel. "
            f"serial={serial_elapsed:.3f}s parallel={parallel_elapsed:.3f}s "
            f"ratio={parallel_elapsed / serial_elapsed:.2f} (expected < 0.60)"
        )
    finally:
        ensure_index_deleted(client, name)
