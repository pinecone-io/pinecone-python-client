from pinecone.grpc.query_results_aggregator import QueryResultsAggregator


class TestQueryResultsAggregator:
    def test_empty_results(self):
        aggregator = QueryResultsAggregator(top_k=3)
        results = aggregator.get_results()
        assert results.usage.read_units == 0
        assert len(results.matches) == 0

    def test_keeps_running_usage_total(self):
        aggregator = QueryResultsAggregator(top_k=3)

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

    def test_inserting_duplicate_scores_stable_ordering(self):
        aggregator = QueryResultsAggregator(top_k=5)

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

    # def test_returns_topk(self):
    #     aggregator = QueryResultsAggregator(top_k=5)

    #     results1 = QueryResponse(
    #         matches=[
    #             ScoredVector(id="1", score=0.1, vector=[]),
    #             ScoredVector(id="2", score=0.11, vector=[]),
    #             ScoredVector(id="3", score=0.12, vector=[]),
    #             ScoredVector(id="4", score=0.13, vector=[]),
    #             ScoredVector(id="5", score=0.14, vector=[]),
    #         ],
    #         usage=Usage(read_units=5)
    #     )
    #     aggregator.add_results(results1)

    #     results2 = QueryResponse(
    #         matches=[
    #             ScoredVector(id="7", score=0.101, vector=[]),
    #             ScoredVector(id="8", score=0.102, vector=[]),
    #             ScoredVector(id="9", score=0.121, vector=[]),
    #             ScoredVector(id="10", score=0.2, vector=[]),
    #             ScoredVector(id="11", score=0.4, vector=[]),
    #         ],
    #         usage=Usage(read_units=7)
    #     )
    #     aggregator.add_results(results2)

    #     combined = aggregator.get_results()
    #     assert len(combined.matches) == 5
    #     assert combined.matches[0].id == "1" # 0.1
    #     assert combined.matches[1].id == "7" # 0.101
    #     assert combined.matches[2].id == "8" # 0.102
    #     assert combined.matches[3].id == "3" # 0.12
    #     assert combined.matches[4].id == "9" # 0.121


class TestQueryResultsAggregatorOutputUX:
    def test_can_interact_with_attributes(self):
        aggregator = QueryResultsAggregator(top_k=1)
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
                }
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
        aggregator = QueryResultsAggregator(top_k=1)
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
                }
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
        aggregator = QueryResultsAggregator(top_k=3)
        results = aggregator.get_results()
        print(results)
        capsys.readouterr()

    def test_can_print_results_containing_None_without_error(self, capsys):
        aggregator = QueryResultsAggregator(top_k=3)
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
        aggregator = QueryResultsAggregator(top_k=4)
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
        aggregator = QueryResultsAggregator(top_k=2)
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
                }
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns1",
        }
        aggregator.add_results(results1)

        results1 = {
            "matches": [
                {
                    "id": "2",
                    "score": 0.4,
                    "values": [0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
                    "sparse_values": {"indices": [1, 2], "values": [0.2, 0.4]},
                    "metadata": {"boolean": True, "nullish": None},
                }
            ],
            "usage": {"readUnits": 5},
            "namespace": "ns2",
        }
        aggregator.add_results(results1)
        results = aggregator.get_results()
        print(results)
        capsys.readouterr()
