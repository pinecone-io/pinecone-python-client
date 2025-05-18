import json
from pinecone import Pinecone
from ..helpers import load_fixture


def upsert(idx, vectors):
    resp = idx.upsert(vectors=vectors, batch_size=25)
    return resp


class TestUpsertSparseVectors:
    def test_upsert_n100_dim768_dict_vectors(self, benchmark, mocker):
        vectors = load_fixture("sparse_100.parquet")
        vectors = [
            {
                "id": v["id"],
                "values": v["values"],
                "sparse_values": {"indices": v["sparse_indices"], "values": v["sparse_values"]},
                "metadata": v["metadata"],
            }
            for v in vectors
        ]
        pc = Pinecone(api_key="fake_api_key")
        idx = pc.Index(host="https://fakehost.pinecone.io")

        # Mock the request method
        mock_request = mocker.Mock()
        response = mocker.Mock()
        response.configure_mock(
            status=200,
            headers={"content-type": "application/json"},
            getheaders=mocker.Mock(return_value={"content-type": "application/json"}),
            data=json.dumps({"upsertedCount": 25}).encode("utf-8"),
            raise_for_status=mocker.Mock(),
        )
        mock_request.return_value = response
        idx._vector_api.api_client.rest_client.pool_manager.request = mock_request

        # Call the benchmark with thresholds
        result = benchmark(upsert, idx, vectors)

        assert result.upserted_count == 100
