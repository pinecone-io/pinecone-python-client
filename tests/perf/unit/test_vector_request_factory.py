import pytest
from pinecone.db_data.request_factory import IndexRequestFactory
from ..helpers import load_fixture


def build_upsert_request(vector_data, _check_type):
    IndexRequestFactory.upsert_request(
        vectors=vector_data, namespace="ns1", _check_type=_check_type
    )


class TestVectorRequestFactoryPerf:
    @pytest.mark.parametrize("check_type", [True, False])
    def test_upsert_request_dense_100_768_dict(self, benchmark, check_type):
        vectors = load_fixture("dense_100_768.parquet")
        benchmark(build_upsert_request, vectors, check_type)

    @pytest.mark.parametrize("check_type", [True, False])
    def test_upsert_request_sparse_100_dict(self, benchmark, check_type):
        vectors = load_fixture("sparse_100.parquet")
        vectors = [
            {
                "id": row["id"],
                "values": row["values"],
                "sparse_values": {"indices": row["sparse_indices"], "values": row["sparse_values"]},
                "metadata": row["metadata"],
            }
            for row in vectors
        ]
        benchmark(build_upsert_request, vectors, check_type)
