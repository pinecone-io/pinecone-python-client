from pinecone import Pinecone
from .helpers import load_fixture


def upsert(idx, vectors):
    idx.upsert(vectors=vectors, batch_size=25)


class TestUpsertPerf:
    def test_upsert_100_768(self, benchmark):
        vectors = load_fixture("dense_100_768.parquet")
        pc = Pinecone()
        index_name = "perf-test"
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                metric="cosine",
                dimension=768,
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            )
        idx = pc.Index(name=index_name)
        benchmark(upsert, idx, vectors)
