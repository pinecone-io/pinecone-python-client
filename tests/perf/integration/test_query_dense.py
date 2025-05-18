import json
import numpy as np
from pinecone import Pinecone
from ..helpers import load_fixture


def run_query(idx, query_vector, include_values, include_metadata):
    resp = idx.query(
        vector=query_vector,
        top_k=100,
        include_values=include_values,
        include_metadata=include_metadata,
    )
    return resp


def fake_results(i):
    matches = load_fixture(f"query_matches_{i}_100_768.parquet")
    # Convert NumPy arrays to Python lists for JSON serialization
    for match in matches:
        if isinstance(match.get("values"), np.ndarray):
            match["values"] = match["values"].tolist()
    return {"results": [], "matches": matches, "namespace": f"ns{i}", "usage": {"readUnits": 1}}


class TestQueryDenseVectors:
    def test_query_dim768_topk100(self, benchmark, mocker):
        pc = Pinecone(api_key="fake_api_key")
        idx = pc.Index(host="https://fakehost.pinecone.io")

        response_dict = fake_results(0)

        # Mock the request method
        mock_request = mocker.Mock()
        response = mocker.Mock()
        response.configure_mock(
            status=200,
            headers={"content-type": "application/json"},
            getheaders=mocker.Mock(return_value={"content-type": "application/json"}),
            data=json.dumps(response_dict).encode("utf-8"),
            raise_for_status=mocker.Mock(),
        )
        mock_request.return_value = response
        idx._vector_api.api_client.rest_client.pool_manager.request = mock_request

        query_vector = load_fixture("dense_100_768.parquet")
        query_vector = query_vector[0]["values"].tolist()

        # Call the benchmark with thresholds
        result = benchmark(
            run_query, idx, query_vector, include_values=False, include_metadata=False
        )

        assert result.results is None
        assert len(result.matches) == 100
        assert result.usage.read_units == 1

    def test_query_dim768_topk100_include_values(self, benchmark, mocker):
        pc = Pinecone(api_key="fake_api_key")
        idx = pc.Index(host="https://fakehost.pinecone.io")

        response_dict = fake_results(0)
        dense_vectors = load_fixture("dense_100_768.parquet")
        for i, m in enumerate(response_dict["matches"]):
            m["values"] = dense_vectors[i]["values"].tolist()
            m["metadata"] = dense_vectors[i]["metadata"]

        # Mock the request method
        mock_request = mocker.Mock()
        response = mocker.Mock()
        response.configure_mock(
            status=200,
            headers={"content-type": "application/json"},
            getheaders=mocker.Mock(return_value={"content-type": "application/json"}),
            data=json.dumps(response_dict).encode("utf-8"),
            raise_for_status=mocker.Mock(),
        )
        mock_request.return_value = response
        idx._vector_api.api_client.rest_client.pool_manager.request = mock_request

        query_vector = dense_vectors[0]["values"].tolist()

        # Call the benchmark with thresholds
        result = benchmark(run_query, idx, query_vector, include_values=True, include_metadata=True)

        assert result.results is None
        assert len(result.matches) == 100
        assert len(result.matches[0]["values"]) == 768
        assert result.matches[0]["metadata"] is not None
        assert result.usage.read_units == 1
