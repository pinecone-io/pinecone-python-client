from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import UpdateRequest, UpdateResponse
from pinecone.grpc.utils import dict_to_proto_struct


class TestGrpcIndexUpdate:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo.pinecone.io")
        self.index = GRPCIndex(
            config=self.config, index_name="example-name", _endpoint_override="test-endpoint"
        )

    def test_update_byIdAnValues_updateByIdAndValues(self, mocker, vals1):
        mock_response = UpdateResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.update(id="vec1", values=vals1, namespace="ns", timeout=30)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(id="vec1", values=vals1, namespace="ns"),
            timeout=30,
        )

    def test_update_byIdAnValuesAsync_updateByIdAndValuesAsync(self, mocker, vals1):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.update(id="vec1", values=vals1, namespace="ns", timeout=30, async_req=True)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update.future,
            UpdateRequest(id="vec1", values=vals1, namespace="ns"),
            timeout=30,
        )

    def test_update_byIdAnValuesAndMetadata_updateByIdAndValuesAndMetadata(
        self, mocker, vals1, md1
    ):
        mock_response = UpdateResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.update("vec1", values=vals1, set_metadata=md1)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(id="vec1", values=vals1, set_metadata=dict_to_proto_struct(md1)),
            timeout=None,
        )
