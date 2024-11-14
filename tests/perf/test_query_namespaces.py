import time
import random
import pytest
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC

latencies = []


def call_n_threads(index):
    query_vec = [random.random() for i in range(1024)]
    start = time.time()
    combined_results = index.query_namespaces(
        vector=query_vec,
        namespaces=["ns1", "ns2", "ns3", "ns4"],
        include_values=False,
        include_metadata=True,
        filter={"publication_date": {"$eq": "Last3Months"}},
        top_k=1000,
    )
    finish = time.time()
    # print(f"Query took {finish-start} seconds")
    latencies.append(finish - start)

    return combined_results


class TestQueryNamespacesRest:
    @pytest.mark.parametrize("n_threads", [4])
    def test_query_namespaces_grpc(self, benchmark, n_threads):
        pc = PineconeGRPC()
        index = pc.Index(
            host="jen1024-dojoi3u.svc.apw5-4e34-81fa.pinecone.io", pool_threads=n_threads
        )
        benchmark.pedantic(call_n_threads, (index,), rounds=10, warmup_rounds=1, iterations=5)

    @pytest.mark.parametrize("n_threads", [4])
    def test_query_namespaces_rest(self, benchmark, n_threads):
        pc = Pinecone()
        index = pc.Index(
            host="jen1024-dojoi3u.svc.apw5-4e34-81fa.pinecone.io",
            pool_threads=n_threads,
            connection_pool_maxsize=20,
        )
        benchmark.pedantic(call_n_threads, (index,), rounds=10, warmup_rounds=1, iterations=5)
