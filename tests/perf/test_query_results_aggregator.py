import random
from pinecone.data.query_results_aggregator import QueryResultsAggregator


def fake_results(i):
    matches = [
        {"id": f"id{i}", "score": random.random(), "values": [random.random() for _ in range(768)]}
        for _ in range(1000)
    ]
    matches.sort(key=lambda x: x["score"], reverse=True)
    return {"namespace": f"ns{i}", "matches": matches}


def aggregate_results(responses):
    ag = QueryResultsAggregator(1000)
    for response in responses:
        ag.add_results(response)
    return ag.get_results()


class TestQueryResultsAggregatorPerf:
    def test_my_stuff(self, benchmark):
        responses = [fake_results(i) for i in range(10)]
        benchmark(aggregate_results, responses)
