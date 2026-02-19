import pytest

from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import QueryRequest, QueryResponse
from pinecone.grpc.utils import dict_to_proto_struct


class TestGrpcIndexQuery:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo.pinecone.io")
        self.index = GRPCIndex(
            config=self.config, index_name="example-name", _endpoint_override="test-endpoint"
        )

    def test_query_byVectorNoFilter_queryVectorNoFilter(self, mocker, vals1):
        mock_response = QueryResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.query(top_k=10, vector=vals1)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Query, QueryRequest(top_k=10, vector=vals1), timeout=None
        )

    def test_query_byVectorWithFilter_queryVectorWithFilter(self, mocker, vals1, filter1):
        mock_response = QueryResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.query(top_k=10, vector=vals1, filter=filter1, namespace="ns", timeout=10)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(
                top_k=10, vector=vals1, filter=dict_to_proto_struct(filter1), namespace="ns"
            ),
            timeout=10,
        )

    def test_query_byVecId_queryByVecId(self, mocker):
        mock_response = QueryResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.query(top_k=10, id="vec1", include_metadata=True, include_values=False)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(top_k=10, id="vec1", include_metadata=True, include_values=False),
            timeout=None,
        )

    def test_query_rejects_both_id_and_vector(self):
        with pytest.raises(ValueError, match="Cannot specify both `id` and `vector`"):
            self.index.query(top_k=10, id="vec1", vector=[1, 2, 3])

    def test_query_with_scan_factor_forwarded_to_proto(self, mocker, vals1):
        mock_response = QueryResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.query(top_k=10, vector=vals1, scan_factor=2.0)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(top_k=10, vector=vals1, scan_factor=2.0),
            timeout=None,
        )

    def test_query_with_max_candidates_forwarded_to_proto(self, mocker, vals1):
        mock_response = QueryResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.query(top_k=10, vector=vals1, max_candidates=500)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(top_k=10, vector=vals1, max_candidates=500),
            timeout=None,
        )

    def test_query_with_scan_factor_and_max_candidates_forwarded_to_proto(self, mocker, vals1):
        mock_response = QueryResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.query(top_k=10, vector=vals1, scan_factor=1.5, max_candidates=1000)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(top_k=10, vector=vals1, scan_factor=1.5, max_candidates=1000),
            timeout=None,
        )

    def test_query_async_with_scan_factor_and_max_candidates_forwarded_to_proto(
        self, mocker, vals1
    ):
        mock_future = mocker.MagicMock()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_future, None))
        self.index.query(
            top_k=10, vector=vals1, scan_factor=0.5, max_candidates=200, async_req=True
        )
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Query.future,
            QueryRequest(top_k=10, vector=vals1, scan_factor=0.5, max_candidates=200),
            timeout=None,
        )


class TestGrpcIndexQueryNamespaces:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo.pinecone.io")
        self.index = GRPCIndex(
            config=self.config, index_name="example-name", _endpoint_override="test-endpoint"
        )

    def test_query_namespaces_forwards_scan_factor_and_max_candidates(self, mocker, vals1):
        mock_response = QueryResponse()
        mocker.patch.object(self.index, "_query", return_value=(mock_response, None))
        self.index.query_namespaces(
            vector=vals1,
            namespaces=["ns1", "ns2"],
            metric="cosine",
            top_k=5,
            scan_factor=3.0,
            max_candidates=750,
        )
        calls = self.index._query.call_args_list
        assert len(calls) == 2
        for call in calls:
            assert call.kwargs["scan_factor"] == 3.0
            assert call.kwargs["max_candidates"] == 750
