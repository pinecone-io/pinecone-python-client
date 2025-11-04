from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import FetchRequest


class TestGrpcIndexFetch:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo.pinecone.io")
        self.index = GRPCIndex(
            config=self.config, index_name="example-name", _endpoint_override="test-endpoint"
        )

    def test_fetch_byIds_fetchByIds(self, mocker):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.fetch(["vec1", "vec2"])
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Fetch, FetchRequest(ids=["vec1", "vec2"]), timeout=None
        )

    def test_fetch_byIdsAndNS_fetchByIdsAndNS(self, mocker):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.fetch(["vec1", "vec2"], namespace="ns", timeout=30)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Fetch, FetchRequest(ids=["vec1", "vec2"], namespace="ns"), timeout=30
        )
