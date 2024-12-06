from pinecone.data.query_results_aggregator import (
    QueryResultsAggregator,
    QueryResultsAggregatorInvalidTopKError,
)
import random
import pytest


class TestQueryResultsAggregator_EuclideanIndex:
    def test_keeps_running_usage_total(self):
        aggregator = QueryResultsAggregator(top_k=3, metric="euclidean")

        results1 = {
            "matches": [
                {"id": "1", "score": 0.1, "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
                {"id": "2", "score": 0.11, "values": []},
                {"id": "3", "score": 0.12, "values": []},
                {"id": "4", "score": 0.13, "values": []},
                {"id": "5", "score": 0.14, "values": []},
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)

        results2 = {
            "matches": [
                {"id": "7", "score": 0.101, "values": []},
                {"id": "8", "score": 0.111, "values": []},
                {"id": "9", "score": 0.12, "values": []},
                {"id": "10", "score": 0.13, "values": []},
                {"id": "11", "score": 0.14, "values": []},
            ],
            "usage": {"readUnits": 7},
            "namespace": "ns2",
        }
        aggregator.add_results(results2)

        results = aggregator.get_results()
        assert results.usage.read_units == 12
        assert len(results.matches) == 3
        assert results.matches[0].id == "1"  # 0.1
        assert results.matches[1].id == "7"  # 0.101
        assert results.matches[2].id == "2"  # 0.11

        # Bracket-style accessor
        assert results["usage"]["read_units"] == results.usage.read_units
        assert results["matches"][0]["id"] == results.matches[0].id

        # Get-style accessor
        assert results.get("matches", []) == results.matches
        assert results.get("usage", {}).get("read_units") == results.usage.read_units

    def test_inserting_duplicate_scores_stable_ordering(self):
        aggregator = QueryResultsAggregator(top_k=5, metric="euclidean")

        results1 = {
            "matches": [
                {"id": "1", "score": 0.11, "values": []},
                {"id": "3", "score": 0.11, "values": []},
                {"id": "2", "score": 0.11, "values": []},
                {"id": "4", "score": 0.22, "values": []},
                {"id": "5", "score": 0.22, "values": []},
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)

        results2 = {
            "matches": [
                {"id": "6", "score": 0.11, "values": []},
                {"id": "7", "score": 0.22, "values": []},
                {"id": "8", "score": 0.22, "values": []},
                {"id": "9", "score": 0.22, "values": []},
                {"id": "10", "score": 0.22, "values": []},
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns2",
        }
        aggregator.add_results(results2)

        results = aggregator.get_results()
        assert results.usage.read_units == 10
        assert len(results.matches) == 5
        assert results.matches[0].id == "1"  # 0.11
        assert results.matches[0].namespace == "ns1"
        assert results.matches[1].id == "3"  # 0.11
        assert results.matches[1].namespace == "ns1"
        assert results.matches[2].id == "2"  # 0.11
        assert results.matches[2].namespace == "ns1"
        assert results.matches[3].id == "6"  # 0.11
        assert results.matches[3].namespace == "ns2"
        assert results.matches[4].id == "4"  # 0.22
        assert results.matches[4].namespace == "ns1"

    def test_still_correct_with_early_return(self):
        aggregator = QueryResultsAggregator(top_k=5, metric="euclidean")

        results1 = {
            "matches": [
                {"id": "1", "score": 0.1, "values": []},
                {"id": "2", "score": 0.11, "values": []},
                {"id": "3", "score": 0.12, "values": []},
                {"id": "4", "score": 0.13, "values": []},
                {"id": "5", "score": 0.14, "values": []},
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)

        results2 = {
            "matches": [
                {"id": "6", "score": 0.10, "values": []},
                {"id": "7", "score": 0.101, "values": []},
                {"id": "8", "score": 0.12, "values": []},
                {"id": "9", "score": 0.13, "values": []},
                {"id": "10", "score": 0.14, "values": []},
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns2",
        }
        aggregator.add_results(results2)

        results = aggregator.get_results()
        assert results.usage.read_units == 10
        assert len(results.matches) == 5
        assert results.matches[0].id == "1"
        assert results.matches[1].id == "6"
        assert results.matches[2].id == "7"
        assert results.matches[3].id == "2"
        assert results.matches[4].id == "3"

    def test_fewer_results_than_topk(self):
        aggregator = QueryResultsAggregator(top_k=1000, metric="euclidean")
        matches1 = [
            {"id": f"id{i}", "score": random.random(), "values": []} for i in range(1, 1000)
        ]
        matches1.sort(key=lambda x: x["score"], reverse=False)

        matches2 = [
            {"id": f"id{i}", "score": random.random(), "values": []} for i in range(1001, 2000)
        ]
        matches2.sort(key=lambda x: x["score"], reverse=False)

        matches3 = [
            {"id": f"id{i}", "score": random.random(), "values": []} for i in range(2001, 3000)
        ]
        matches3.sort(key=lambda x: x["score"], reverse=False)

        matches4 = [
            {"id": f"id{i}", "score": random.random(), "values": []} for i in range(3001, 4000)
        ]
        matches4.sort(key=lambda x: x["score"], reverse=False)

        matches5 = [
            {"id": f"id{i}", "score": random.random(), "values": []} for i in range(4001, 5000)
        ]
        matches5.sort(key=lambda x: x["score"], reverse=False)

        results1 = {"matches": matches1, "namespace": "ns1", "usage": {"readUnits": 5}}
        results2 = {"matches": matches2, "namespace": "ns2", "usage": {"readUnits": 5}}
        results3 = {"matches": matches3, "namespace": "ns3", "usage": {"readUnits": 5}}
        results4 = {"matches": matches4, "namespace": "ns4", "usage": {"readUnits": 5}}
        results5 = {"matches": matches5, "namespace": "ns5", "usage": {"readUnits": 5}}

        aggregator.add_results(results1)
        aggregator.add_results(results2)
        aggregator.add_results(results3)
        aggregator.add_results(results4)
        aggregator.add_results(results5)

        merged_matches = matches1 + matches2 + matches3 + matches4 + matches5
        merged_matches.sort(key=lambda x: x["score"], reverse=False)

        results = aggregator.get_results()
        assert results.usage.read_units == 25
        assert len(results.matches) == 1000
        assert results.matches[0].score == merged_matches[0]["score"]
        assert results.matches[1].score == merged_matches[1]["score"]
        assert results.matches[2].score == merged_matches[2]["score"]
        assert results.matches[3].score == merged_matches[3]["score"]
        assert results.matches[4].score == merged_matches[4]["score"]


class TestQueryResultsAggregator_CosineOrDotproductIndex:
    def test_inserting_duplicate_scores_stable_ordering(self):
        aggregator = QueryResultsAggregator(top_k=6, metric="cosine")

        results1 = {
            "matches": [
                {"id": "1", "score": 0.94, "values": []},
                {"id": "3", "score": 0.94, "values": []},
                {"id": "2", "score": 0.94, "values": []},
                {"id": "4", "score": 0.88, "values": []},
                {"id": "5", "score": 0.88, "values": []},
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)

        results2 = {
            "matches": [
                {"id": "6", "score": 0.93, "values": []},
                {"id": "7", "score": 0.93, "values": []},
                {"id": "8", "score": 0.85, "values": []},
                {"id": "9", "score": 0.85, "values": []},
                {"id": "10", "score": 0.80, "values": []},
            ],
            "usage": {"readUnits": 8},
            "namespace": "ns2",
        }
        aggregator.add_results(results2)

        results = aggregator.get_results()
        assert results.usage.read_units == 13
        assert len(results.matches) == 6
        assert results.matches[0].id == "1"
        assert results.matches[0].namespace == "ns1"
        assert results.matches[1].id == "3"
        assert results.matches[1].namespace == "ns1"
        assert results.matches[2].id == "2"
        assert results.matches[2].namespace == "ns1"
        assert results.matches[3].id == "6"
        assert results.matches[3].namespace == "ns2"
        assert results.matches[4].id == "7"
        assert results.matches[4].namespace == "ns2"
        assert results.matches[5].id == "4"
        assert results.matches[5].namespace == "ns1"

    @pytest.mark.parametrize("index_metric", ["cosine", "dotproduct"])
    def test_correctly_handles_dotproduct_and_cosine_metric(self, index_metric):
        # For this index metric, the higher the score, the more similar the vectors are.
        aggregator = QueryResultsAggregator(top_k=3, metric=index_metric)

        desc_results1 = {
            "matches": [
                {"id": "1", "score": 0.9, "values": []},
                {"id": "2", "score": 0.8, "values": []},
                {"id": "3", "score": 0.7, "values": []},
                {"id": "4", "score": 0.6, "values": []},
                {"id": "5", "score": 0.5, "values": []},
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(desc_results1)

        results2 = {
            "matches": [
                {"id": "7", "score": 0.77, "values": []},
                {"id": "8", "score": 0.88, "values": []},
                {"id": "9", "score": 0.99, "values": []},
                {"id": "10", "score": 0.1010, "values": []},
                {"id": "11", "score": 0.1111, "values": []},
            ],
            "usage": {"readUnits": 7},
            "namespace": "ns2",
        }
        aggregator.add_results(results2)

        results = aggregator.get_results()
        assert results.usage.read_units == 12
        assert len(results.matches) == 3
        assert results.matches[0].id == "9"  # 0.99
        assert results.matches[1].id == "1"  # 0.9
        assert results.matches[2].id == "8"  # 0.88

    def test_lots_of_results(self):
        aggregator = QueryResultsAggregator(top_k=1000, metric="dotproduct")
        matches1 = [
            {"id": f"id{i}", "score": random.random(), "values": []} for i in range(1, 1000)
        ]
        matches1.sort(key=lambda x: x["score"], reverse=True)

        matches2 = [
            {"id": f"id{i}", "score": random.random(), "values": []} for i in range(1001, 2000)
        ]
        matches2.sort(key=lambda x: x["score"], reverse=True)

        matches3 = [
            {"id": f"id{i}", "score": random.random(), "values": []} for i in range(2001, 3000)
        ]
        matches3.sort(key=lambda x: x["score"], reverse=True)

        matches4 = [
            {"id": f"id{i}", "score": random.random(), "values": []} for i in range(3001, 4000)
        ]
        matches4.sort(key=lambda x: x["score"], reverse=True)

        matches5 = [
            {"id": f"id{i}", "score": random.random(), "values": []} for i in range(4001, 5000)
        ]
        matches5.sort(key=lambda x: x["score"], reverse=True)

        results1 = {"matches": matches1, "namespace": "ns1", "usage": {"readUnits": 5}}
        results2 = {"matches": matches2, "namespace": "ns2", "usage": {"readUnits": 5}}
        results3 = {"matches": matches3, "namespace": "ns3", "usage": {"readUnits": 5}}
        results4 = {"matches": matches4, "namespace": "ns4", "usage": {"readUnits": 5}}
        results5 = {"matches": matches5, "namespace": "ns5", "usage": {"readUnits": 5}}

        aggregator.add_results(results1)
        aggregator.add_results(results2)
        aggregator.add_results(results3)
        aggregator.add_results(results4)
        aggregator.add_results(results5)

        merged_matches = matches1 + matches2 + matches3 + matches4 + matches5
        merged_matches.sort(key=lambda x: x["score"], reverse=True)

        results = aggregator.get_results()
        assert results.usage.read_units == 25
        assert len(results.matches) == 1000
        assert results.matches[0].score == merged_matches[0]["score"]
        assert results.matches[1].score == merged_matches[1]["score"]
        assert results.matches[2].score == merged_matches[2]["score"]
        assert results.matches[3].score == merged_matches[3]["score"]
        assert results.matches[4].score == merged_matches[4]["score"]


class TestQueryResultsAggregatorOutputUX:
    def test_can_interact_with_attributes(self):
        aggregator = QueryResultsAggregator(top_k=2, metric="euclidean")
        results1 = {
            "matches": [
                {
                    "id": "1",
                    "score": 0.3,
                    "values": [0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
                    "sparse_values": {"indices": [1, 2], "values": [0.2, 0.4]},
                    "metadata": {
                        "hello": "world",
                        "number": 42,
                        "list": [1, 2, 3],
                        "list2": ["foo", "bar"],
                    },
                },
                {"id": "2", "score": 0.4},
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)
        results = aggregator.get_results()
        assert results.usage.read_units == 5
        assert results.matches[0].id == "1"
        assert results.matches[0].namespace == "ns1"
        assert results.matches[0].score == 0.3
        assert results.matches[0].values == [0.31, 0.32, 0.33, 0.34, 0.35, 0.36]

    def test_can_interact_like_dict(self):
        aggregator = QueryResultsAggregator(top_k=3, metric="euclidean")
        results1 = {
            "matches": [
                {
                    "id": "1",
                    "score": 0.3,
                    "values": [0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
                    "sparse_values": {"indices": [1, 2], "values": [0.2, 0.4]},
                    "metadata": {
                        "hello": "world",
                        "number": 42,
                        "list": [1, 2, 3],
                        "list2": ["foo", "bar"],
                    },
                },
                {
                    "id": "2",
                    "score": 0.4,
                    "values": [0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
                    "sparse_values": {"indices": [1, 2], "values": [0.2, 0.4]},
                    "metadata": {
                        "hello": "world",
                        "number": 42,
                        "list": [1, 2, 3],
                        "list2": ["foo", "bar"],
                    },
                },
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)
        results = aggregator.get_results()
        assert results["usage"]["read_units"] == 5
        assert results["matches"][0]["id"] == "1"
        assert results["matches"][0]["namespace"] == "ns1"
        assert results["matches"][0]["score"] == 0.3

    def test_can_print_empty_results_without_error(self, capsys):
        aggregator = QueryResultsAggregator(top_k=3, metric="euclidean")
        results = aggregator.get_results()
        print(results)
        capsys.readouterr()

    def test_can_print_results_containing_None_without_error(self, capsys):
        aggregator = QueryResultsAggregator(top_k=3, metric="euclidean")
        results1 = {
            "matches": [
                {"id": "1", "score": 0.1},
                {"id": "2", "score": 0.11},
                {"id": "3", "score": 0.12},
                {"id": "4", "score": 0.13},
                {"id": "5", "score": 0.14},
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)
        results = aggregator.get_results()
        print(results)
        capsys.readouterr()

    def test_can_print_results_containing_short_vectors(self, capsys):
        aggregator = QueryResultsAggregator(top_k=4, metric="euclidean")
        results1 = {
            "matches": [
                {"id": "1", "score": 0.1, "values": [0.31]},
                {"id": "2", "score": 0.11, "values": [0.31, 0.32]},
                {"id": "3", "score": 0.12, "values": [0.31, 0.32, 0.33]},
                {"id": "3", "score": 0.12, "values": [0.31, 0.32, 0.33, 0.34]},
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)
        results = aggregator.get_results()
        print(results)
        capsys.readouterr()

    def test_can_print_complete_results_without_error(self, capsys):
        aggregator = QueryResultsAggregator(top_k=2, metric="euclidean")
        results1 = {
            "matches": [
                {
                    "id": "1",
                    "score": 0.3,
                    "values": [0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
                    "sparse_values": {"indices": [1, 2], "values": [0.2, 0.4]},
                    "metadata": {
                        "hello": "world",
                        "number": 42,
                        "list": [1, 2, 3],
                        "list2": ["foo", "bar"],
                    },
                },
                {
                    "id": "2",
                    "score": 0.4,
                    "values": [0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
                    "sparse_values": {"indices": [1, 2], "values": [0.2, 0.4]},
                    "metadata": {"boolean": True, "nullish": None},
                },
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)
        results = aggregator.get_results()
        print(results)
        capsys.readouterr()


class TestQueryAggregatorEdgeCases:
    def test_topK_too_small(self):
        with pytest.raises(QueryResultsAggregatorInvalidTopKError):
            QueryResultsAggregator(top_k=0, metric="euclidean")

    def test_results_never_added(self):
        aggregator = QueryResultsAggregator(top_k=3, metric="euclidean")
        results = aggregator.get_results()
        assert results is not None
        assert results.usage.read_units == 0
        assert len(results.matches) == 0

    @pytest.mark.parametrize("index_metric", ["euclidean", "cosine", "dotproduct"])
    def test_tie_breaking(self, index_metric):
        aggregator = QueryResultsAggregator(top_k=3, metric=index_metric)
        results1 = {
            "matches": [{"id": "5", "score": 0.9}, {"id": "2", "score": 0.9}],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }

        results2 = {
            "matches": [{"id": "3", "score": 0.9}, {"id": "4", "score": 0.9}],
            "usage": {"readUnits": 5},
            "namespace": "ns2",
        }
        aggregator.add_results(results1)
        aggregator.add_results(results2)
        results = aggregator.get_results()
        assert results.usage.read_units == 10
        assert len(results.matches) == 3

        # Maintains order results were added when resolving ties
        assert results.matches[0].id == "5"
        assert results.matches[0].namespace == "ns1"
        assert results.matches[1].id == "2"
        assert results.matches[1].namespace == "ns1"
        assert results.matches[2].id == "3"
        assert results.matches[2].namespace == "ns2"

    def test_empty_results_with_usage(self):
        aggregator = QueryResultsAggregator(top_k=3, metric="euclidean")

        aggregator.add_results({"matches": [], "usage": {"readUnits": 5}, "namespace": "ns1"})
        aggregator.add_results({"matches": [], "usage": {"readUnits": 5}, "namespace": "ns2"})
        aggregator.add_results({"matches": [], "usage": {"readUnits": 5}, "namespace": "ns3"})

        results = aggregator.get_results()
        assert results is not None
        assert results.usage.read_units == 15
        assert len(results.matches) == 0

    def test_exactly_one_result(self):
        aggregator = QueryResultsAggregator(top_k=3, metric="euclidean")
        results1 = {
            "matches": [{"id": "2", "score": 0.01}, {"id": "3", "score": 0.2}],
            "usage": {"readUnits": 5},
            "namespace": "ns2",
        }
        aggregator.add_results(results1)

        results2 = {
            "matches": [{"id": "1", "score": 0.1}],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results2)
        results = aggregator.get_results()
        assert results.usage.read_units == 10
        assert len(results.matches) == 3
        assert results.matches[0].id == "2"
        assert results.matches[0].namespace == "ns2"
        assert results.matches[0].score == 0.01
        assert results.matches[1].id == "1"
        assert results.matches[1].namespace == "ns1"
        assert results.matches[1].score == 0.1
        assert results.matches[2].id == "3"
        assert results.matches[2].namespace == "ns2"
        assert results.matches[2].score == 0.2

    def test_two_result_sets_with_single_result(self):
        aggregator = QueryResultsAggregator(top_k=3, metric="euclidean")
        results1 = {
            "matches": [{"id": "1", "score": 0.1}],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)
        results2 = {
            "matches": [{"id": "2", "score": 0.01}],
            "usage": {"readUnits": 5},
            "namespace": "ns2",
        }
        aggregator.add_results(results2)
        results = aggregator.get_results()
        assert results.usage.read_units == 10
        assert len(results.matches) == 2
        assert results.matches[0].id == "2"
        assert results.matches[0].namespace == "ns2"
        assert results.matches[0].score == 0.01
        assert results.matches[1].id == "1"
        assert results.matches[1].namespace == "ns1"
        assert results.matches[1].score == 0.1

    def test_all_empty_results(self):
        aggregator = QueryResultsAggregator(top_k=10, metric="euclidean")

        aggregator.add_results({"matches": [], "usage": {"readUnits": 5}, "namespace": "ns1"})
        aggregator.add_results({"matches": [], "usage": {"readUnits": 5}, "namespace": "ns2"})
        aggregator.add_results({"matches": [], "usage": {"readUnits": 5}, "namespace": "ns3"})

        results = aggregator.get_results()

        assert results.usage.read_units == 15
        assert len(results.matches) == 0

    def test_some_empty_results(self):
        aggregator = QueryResultsAggregator(top_k=10, metric="euclidean")
        results2 = {
            "matches": [{"id": "2", "score": 0.01}, {"id": "3", "score": 0.2}],
            "usage": {"readUnits": 5},
            "namespace": "ns0",
        }
        aggregator.add_results(results2)

        aggregator.add_results({"matches": [], "usage": {"readUnits": 5}, "namespace": "ns1"})
        aggregator.add_results({"matches": [], "usage": {"readUnits": 5}, "namespace": "ns2"})
        aggregator.add_results({"matches": [], "usage": {"readUnits": 5}, "namespace": "ns3"})

        results = aggregator.get_results()

        assert results.usage.read_units == 20
        assert len(results.matches) == 2
