from pinecone.db_data.vector_factory import VectorFactory
from .helpers import load_fixture


def build_vector_objects(vector_data):
    for row in vector_data:
        VectorFactory.build(row)


class TestVectorFactoryPerf:
    def test_vector_factory_100_768_dict(self, benchmark):
        vectors = load_fixture("dense_100_768.parquet")
        benchmark(build_vector_objects, vectors)

    def test_vector_factory_100_768_tuple(self, benchmark):
        vectors = load_fixture("dense_100_768.parquet")
        vectors = [(row["id"], row["values"], row["metadata"]) for row in vectors]
        benchmark(build_vector_objects, vectors)
