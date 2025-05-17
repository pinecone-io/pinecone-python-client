import json
from pinecone import Pinecone
from .helpers import load_fixture


def upsert(idx, vectors):
    idx.upsert(vectors=vectors, batch_size=25)


class TestUpsertPerf:
    def test_upsert_100_768(self, benchmark, mocker):
        vectors = load_fixture("dense_100_768.parquet")
        pc = Pinecone()
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

        # Call the benchmark
        benchmark(upsert, idx, vectors)
