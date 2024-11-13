import pytest
import os
from ..helpers import random_string, poll_stats_for_namespace
from pinecone.data.query_results_aggregator import (
    QueryResultsAggregatorInvalidTopKError,
    QueryResultsAggregregatorNotEnoughResultsError,
)

from pinecone import Vector


@pytest.mark.skipif(
    os.getenv("USE_GRPC") == "true", reason="query_namespaces currently only available via rest"
)
class TestQueryNamespacesRest:
    def test_query_namespaces(self, idx):
        ns_prefix = random_string(5)
        ns1 = f"{ns_prefix}-ns1"
        ns2 = f"{ns_prefix}-ns2"
        ns3 = f"{ns_prefix}-ns3"

        idx.upsert(
            vectors=[
                Vector(id="id1", values=[0.1, 0.2], metadata={"genre": "drama", "key": 1}),
                Vector(id="id2", values=[0.2, 0.3], metadata={"genre": "drama", "key": 2}),
                Vector(id="id3", values=[0.4, 0.5], metadata={"genre": "action", "key": 3}),
                Vector(id="id4", values=[0.6, 0.7], metadata={"genre": "action", "key": 4}),
            ],
            namespace=ns1,
        )
        idx.upsert(
            vectors=[
                Vector(id="id5", values=[0.21, 0.22], metadata={"genre": "drama", "key": 1}),
                Vector(id="id6", values=[0.22, 0.23], metadata={"genre": "drama", "key": 2}),
                Vector(id="id7", values=[0.24, 0.25], metadata={"genre": "action", "key": 3}),
                Vector(id="id8", values=[0.26, 0.27], metadata={"genre": "action", "key": 4}),
            ],
            namespace=ns2,
        )
        idx.upsert(
            vectors=[
                Vector(id="id9", values=[0.31, 0.32], metadata={"genre": "drama", "key": 1}),
                Vector(id="id10", values=[0.32, 0.33], metadata={"genre": "drama", "key": 2}),
                Vector(id="id11", values=[0.34, 0.35], metadata={"genre": "action", "key": 3}),
                Vector(id="id12", values=[0.36, 0.37], metadata={"genre": "action", "key": 4}),
            ],
            namespace=ns3,
        )

        poll_stats_for_namespace(idx, namespace=ns1, expected_count=4)
        poll_stats_for_namespace(idx, namespace=ns2, expected_count=4)
        poll_stats_for_namespace(idx, namespace=ns3, expected_count=4)

        results = idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[ns1, ns2, ns3],
            include_values=True,
            include_metadata=True,
            filter={"genre": {"$eq": "drama"}},
            top_k=100,
        )
        assert len(results.matches) == 6
        assert results.usage.read_units > 0
        for item in results.matches:
            assert item.metadata["genre"] == "drama"
        assert results.matches[0].id == "id1"
        assert results.matches[0].namespace == ns1

        # Using dot-style accessors
        assert results.matches[0].metadata["genre"] == "drama"
        assert results.matches[0].metadata["key"] == 1

        # Using dictionary-style accessors
        assert results.matches[0]["metadata"]["genre"] == "drama"
        assert results.matches[0]["metadata"]["key"] == 1

        # Using .get() accessors
        assert results.get("matches", [])[0].get("metadata", {}).get("genre") == "drama"
        assert results.matches[0].get("metadata", {}) == {"genre": "drama", "key": 1}
        assert results.matches[0].get("metadata", {}).get("genre") == "drama"

        assert results.matches[1].id == "id2"
        assert results.matches[1].namespace == ns1
        assert results.matches[2].id == "id5"
        assert results.matches[2].namespace == ns2

        # Non-existent namespace shouldn't cause any problem
        results2 = idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[ns1, ns2, ns3, f"{ns_prefix}-nonexistent"],
            include_values=True,
            include_metadata=True,
            filter={"genre": {"$eq": "action"}},
            top_k=100,
        )
        assert len(results2.matches) == 6
        assert results2.usage.read_units > 0
        for item in results2.matches:
            assert item.metadata["genre"] == "action"

        # Test with empty filter, top_k greater than number of results
        results3 = idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[ns1, ns2, ns3],
            include_values=True,
            include_metadata=True,
            filter={},
            top_k=100,
        )
        assert len(results3.matches) == 12
        assert results3.usage.read_units > 0

        # Test when all results are filtered out
        results4 = idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[ns1, ns2, ns3],
            include_values=True,
            include_metadata=True,
            filter={"genre": {"$eq": "comedy"}},
            top_k=100,
        )
        assert len(results4.matches) == 0
        assert results4.usage.read_units > 0

        # Test with top_k less than number of results
        results5 = idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[ns1, ns2, ns3],
            include_values=True,
            include_metadata=True,
            filter={},
            top_k=2,
        )
        assert len(results5.matches) == 2

        # Test when all namespaces are non-existent (same as all results filtered / empty)
        results6 = idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[
                f"{ns_prefix}-nonexistent1",
                f"{ns_prefix}-nonexistent2",
                f"{ns_prefix}-nonexistent3",
            ],
            include_values=True,
            include_metadata=True,
            filter={"genre": {"$eq": "comedy"}},
            top_k=2,
        )
        assert len(results6.matches) == 0
        assert results6.usage.read_units > 0

    def test_invalid_top_k(self, idx):
        with pytest.raises(QueryResultsAggregatorInvalidTopKError) as e:
            idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1", "ns2", "ns3"],
                include_values=True,
                include_metadata=True,
                filter={},
                top_k=1,
            )
        assert (
            str(e.value)
            == "Invalid top_k value 1. To aggregate results from multiple queries the top_k must be at least 2."
        )

    def test_unmergeable_results(self, idx):
        ns_prefix = random_string(5)
        ns1 = f"{ns_prefix}-ns1"
        ns2 = f"{ns_prefix}-ns2"

        idx.upsert(
            vectors=[
                Vector(id="id1", values=[0.1, 0.2], metadata={"genre": "drama", "key": 1}),
                Vector(id="id2", values=[0.2, 0.3], metadata={"genre": "drama", "key": 2}),
            ],
            namespace=ns1,
        )
        idx.upsert(
            vectors=[
                Vector(id="id5", values=[0.21, 0.22], metadata={"genre": "drama", "key": 1}),
                Vector(id="id6", values=[0.22, 0.23], metadata={"genre": "drama", "key": 2}),
            ],
            namespace=ns2,
        )

        poll_stats_for_namespace(idx, namespace=ns1, expected_count=2)
        poll_stats_for_namespace(idx, namespace=ns2, expected_count=2)

        with pytest.raises(QueryResultsAggregregatorNotEnoughResultsError) as e:
            idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=[ns1, ns2],
                include_values=True,
                include_metadata=True,
                filter={"key": {"$eq": 1}},
                top_k=2,
            )

        assert (
            str(e.value)
            == "Cannot interpret results without at least two matches. In order to aggregate results from multiple queries, top_k must be greater than 1 in order to correctly infer the similarity metric from scores."
        )

    def test_missing_namespaces(self, idx):
        with pytest.raises(ValueError) as e:
            idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=[],
                include_values=True,
                include_metadata=True,
                filter={},
                top_k=2,
            )
        assert str(e.value) == "At least one namespace must be specified"

        with pytest.raises(ValueError) as e:
            idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=None,
                include_values=True,
                include_metadata=True,
                filter={},
                top_k=2,
            )
        assert str(e.value) == "At least one namespace must be specified"
