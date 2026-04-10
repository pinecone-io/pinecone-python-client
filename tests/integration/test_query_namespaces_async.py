"""Integration tests for advanced query_namespaces operations (async REST).

Phase 3 Tier 5: query-namespaces-filter, query-namespaces-many.
"""
# area tags covered: query-namespaces-filter, query-namespaces-many

from __future__ import annotations

import pytest
import pytest_asyncio

from pinecone import AsyncPinecone
from pinecone.async_client.async_index import AsyncIndex
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.vector import ScoredVector
from tests.integration.conftest import (
    async_cleanup_resource,
    async_poll_until,
    unique_name,
)


# ---------------------------------------------------------------------------
# query-namespaces-filter — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_namespaces_filter_rest_async(async_client: AsyncPinecone) -> None:
    """query_namespaces() with filter applies it per-namespace and returns metadata (REST async)."""
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

        # Populate host cache so async_client.index(name=...) can resolve it
        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        # Upsert comedy + drama vectors into two namespaces
        await idx.upsert(
            vectors=[
                {"id": "qnf-ns1-com1", "values": [0.1, 0.2], "metadata": {"genre": "comedy"}},
                {"id": "qnf-ns1-dra1", "values": [0.9, 0.8], "metadata": {"genre": "drama"}},
            ],
            namespace="qnf-ns1",
        )
        await idx.upsert(
            vectors=[
                {"id": "qnf-ns2-com1", "values": [0.2, 0.3], "metadata": {"genre": "comedy"}},
                {"id": "qnf-ns2-dra1", "values": [0.8, 0.7], "metadata": {"genre": "drama"}},
            ],
            namespace="qnf-ns2",
        )

        # Wait for all vectors in both namespaces to be queryable
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10, namespace="qnf-ns1"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="ns1 vectors queryable (async) before query_namespaces_filter",
        )
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10, namespace="qnf-ns2"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="ns2 vectors queryable (async) before query_namespaces_filter",
        )

        # Call query_namespaces with comedy filter and include_metadata=True
        results = await idx.query_namespaces(
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
        assert len(match_ids & comedy_ids) >= 1
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
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# query-namespaces-many — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_namespaces_many_rest_async(async_client: AsyncPinecone) -> None:
    """query_namespaces() across 5+ namespaces merges and sorts results; ns_usage has entry per namespace (REST async)."""
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

        # Populate host cache so async_client.index(name=...) can resolve it
        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        # Upsert 2 vectors into each of 5 namespaces
        namespaces = [f"qnm-ns-{i}" for i in range(5)]
        for i, ns in enumerate(namespaces):
            base = float(i) / 5.0
            await idx.upsert(
                vectors=[
                    {"id": f"{ns}-v1", "values": [base, 1.0 - base]},
                    {"id": f"{ns}-v2", "values": [1.0 - base, base]},
                ],
                namespace=ns,
            )

        # Wait for each namespace to have both vectors queryable
        for ns in namespaces:
            await async_poll_until(
                query_fn=lambda ns=ns: idx.query(vector=[0.5, 0.5], top_k=10, namespace=ns),
                check_fn=lambda r: len(r.matches) >= 2,
                timeout=120,
                description=f"{ns} vectors queryable (async) before query_namespaces_many",
            )

        # Query across all 5 namespaces at once
        results = await idx.query_namespaces(
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
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )
