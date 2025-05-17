import pytest
from pinecone.db_data.query_results_aggregator import QueryResultsAggregator
from .helpers import load_fixture


def fake_results(i):
    matches = load_fixture(f"query_matches_{i}_100_768.parquet")
    return {"namespace": f"ns{i}", "matches": matches}


def aggregate_results(responses):
    ag = QueryResultsAggregator(100, "cosine")
    for response in responses:
        ag.add_results(response)
    return ag.get_results()


class TestQueryResultsAggregatorPerf:
    @pytest.mark.parametrize("num_namespaces", [1, 5, 10])
    def test_merge_namespaces_with_values(self, benchmark, num_namespaces):
        responses = [fake_results(i) for i in range(num_namespaces)]
        benchmark(aggregate_results, responses)

    @pytest.mark.parametrize("num_namespaces", [1, 5, 10])
    def test_merge_namespaces_without_values(self, benchmark, num_namespaces):
        responses = [fake_results(i) for i in range(num_namespaces)]
        for response in responses:
            response["matches"] = [
                {"id": match["id"], "score": match["score"]} for match in response["matches"]
            ]
        benchmark(aggregate_results, responses)
