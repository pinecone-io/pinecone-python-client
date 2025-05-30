import pytest
from ..helpers import random_string, poll_stats_for_namespace
from pinecone.db_data.query_results_aggregator import QueryResultsAggregatorInvalidTopKError

from pinecone import Vector, SparseValues


@pytest.mark.skip(reason="Sparse indexes are not yet supported")
class TestQueryNamespacesRest_Sparse:
    def test_query_namespaces(self, sparse_idx):
        ns_prefix = random_string(5)
        ns1 = f"{ns_prefix}-ns1"
        ns2 = f"{ns_prefix}-ns2"
        ns3 = f"{ns_prefix}-ns3"

        sparse_idx.upsert(
            vectors=[
                Vector(
                    id="id1",
                    sparse_values=SparseValues(indices=[1, 0, 2], values=[0.1, 0.2, 0.3]),
                    metadata={"genre": "drama", "key": 1},
                ),
                Vector(
                    id="id2",
                    sparse_values=SparseValues(indices=[1, 2, 3], values=[0.2, 0.3, 0.3]),
                    metadata={"genre": "drama", "key": 2},
                ),
                Vector(
                    id="id3",
                    sparse_values=SparseValues(indices=[1, 4, 5], values=[0.4, 0.5, 0.6]),
                    metadata={"genre": "action", "key": 3},
                ),
                Vector(
                    id="id4",
                    sparse_values=SparseValues(indices=[1, 6, 7], values=[0.6, 0.7, 0.7]),
                    metadata={"genre": "action", "key": 4},
                ),
            ],
            namespace=ns1,
        )
        sparse_idx.upsert(
            vectors=[
                Vector(
                    id="id5",
                    sparse_values=SparseValues(indices=[1, 8, 9], values=[0.21, 0.22, 0.23]),
                    metadata={"genre": "drama", "key": 1},
                ),
                Vector(
                    id="id6",
                    sparse_values=SparseValues(indices=[1, 10, 11], values=[0.22, 0.23, 0.24]),
                    metadata={"genre": "drama", "key": 2},
                ),
                Vector(
                    id="id7",
                    sparse_values=SparseValues(indices=[1, 12, 13], values=[0.24, 0.25, 0.26]),
                    metadata={"genre": "action", "key": 3},
                ),
                Vector(
                    id="id8",
                    sparse_values=SparseValues(indices=[1, 14, 15], values=[0.26, 0.27, 0.27]),
                    metadata={"genre": "action", "key": 4},
                ),
            ],
            namespace=ns2,
        )
        sparse_idx.upsert(
            vectors=[
                Vector(
                    id="id9",
                    sparse_values=SparseValues(indices=[1, 16, 17], values=[0.31, 0.32, 0.33]),
                    metadata={"genre": "drama", "key": 1},
                ),
                Vector(
                    id="id10",
                    sparse_values=SparseValues(indices=[1, 18, 19], values=[0.32, 0.33, 0.34]),
                    metadata={"genre": "drama", "key": 2},
                ),
                Vector(
                    id="id11",
                    sparse_values=SparseValues(indices=[1, 20, 21], values=[0.34, 0.35, 0.35]),
                    metadata={"genre": "action", "key": 3},
                ),
                Vector(
                    id="id12",
                    sparse_values=SparseValues(indices=[1, 22, 23], values=[0.36, 0.37, 0.36]),
                    metadata={"genre": "action", "key": 4},
                ),
            ],
            namespace=ns3,
        )

        poll_stats_for_namespace(sparse_idx, namespace=ns1, expected_count=4)
        poll_stats_for_namespace(sparse_idx, namespace=ns2, expected_count=4)
        poll_stats_for_namespace(sparse_idx, namespace=ns3, expected_count=4)

        results = sparse_idx.query_namespaces(
            sparse_vector=SparseValues(indices=[1], values=[24.5]),
            metric="dotproduct",
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
        results2 = sparse_idx.query_namespaces(
            sparse_vector=SparseValues(indices=[1], values=[24.5]),
            namespaces=[ns1, ns2, ns3, f"{ns_prefix}-nonexistent"],
            metric="dotproduct",
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
        results3 = sparse_idx.query_namespaces(
            sparse_vector=SparseValues(indices=[1], values=[24.5]),
            namespaces=[ns1, ns2, ns3],
            metric="dotproduct",
            include_values=True,
            include_metadata=True,
            filter={},
            top_k=100,
        )
        assert len(results3.matches) == 12
        assert results3.usage.read_units > 0

        # Test when all results are filtered out
        results4 = sparse_idx.query_namespaces(
            sparse_vector=SparseValues(indices=[1000], values=[24.5]),
            namespaces=[ns1, ns2, ns3],
            metric="dotproduct",
            include_values=True,
            include_metadata=True,
            filter={"genre": {"$eq": "comedy"}},
            top_k=100,
        )
        assert len(results4.matches) == 0
        assert results4.usage.read_units > 0

        # Test with top_k less than number of results
        results5 = sparse_idx.query_namespaces(
            sparse_vector=SparseValues(indices=[1], values=[24.5]),
            namespaces=[ns1, ns2, ns3],
            metric="dotproduct",
            include_values=True,
            include_metadata=True,
            filter={},
            top_k=2,
        )
        assert len(results5.matches) == 2

        # Test when all namespaces are non-existent (same as all results filtered / empty)
        results6 = sparse_idx.query_namespaces(
            sparse_vector=SparseValues(indices=[1], values=[24.5]),
            namespaces=[
                f"{ns_prefix}-nonexistent1",
                f"{ns_prefix}-nonexistent2",
                f"{ns_prefix}-nonexistent3",
            ],
            metric="dotproduct",
            include_values=True,
            include_metadata=True,
            filter={"genre": {"$eq": "comedy"}},
            top_k=2,
        )
        assert len(results6.matches) == 0
        assert results6.usage.read_units > 0

    def test_invalid_top_k(self, sparse_idx):
        with pytest.raises(QueryResultsAggregatorInvalidTopKError) as e:
            sparse_idx.query_namespaces(
                sparse_vector=SparseValues(indices=[1], values=[24.5]),
                namespaces=["ns1", "ns2", "ns3"],
                metric="dotproduct",
                include_values=True,
                include_metadata=True,
                filter={},
                top_k=1,
            )
        assert (
            str(e.value)
            == "Invalid top_k value 1. To aggregate results from multiple queries the top_k must be at least 2."
        )

    def test_missing_namespaces(self, sparse_idx):
        with pytest.raises(ValueError) as e:
            sparse_idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=[],
                metric="dotproduct",
                include_values=True,
                include_metadata=True,
                filter={},
                top_k=2,
            )
        assert str(e.value) == "At least one namespace must be specified"

        with pytest.raises(ValueError) as e:
            sparse_idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=None,
                metric="dotproduct",
                include_values=True,
                include_metadata=True,
                filter={},
                top_k=2,
            )
        assert str(e.value) == "At least one namespace must be specified"
