from pinecone.db_data.sparse_values_factory import SparseValuesFactory
from .helpers import load_fixture


def build_sparse_values_objects(vectors):
    for row in vectors:
        SparseValuesFactory.build(row)


class TestSparseVectorFactoryPerf:
    def test_sparse_vector_factory_100_dict(self, benchmark):
        sparse_values_data = load_fixture("sparse_100.parquet")
        vectors = [
            {"indices": row["sparse_indices"], "values": row["sparse_values"]}
            for row in sparse_values_data
        ]
        benchmark(build_sparse_values_objects, vectors)
