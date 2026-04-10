"""Integration tests for advanced query_namespaces operations (sync REST).

Phase 3 Tier 5: query-namespaces-filter, query-namespaces-many.
"""
# area tags covered: query-namespaces-filter, query-namespaces-many

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.vector import ScoredVector
from tests.integration.conftest import cleanup_resource, poll_until, unique_name


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
