from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import UpdateRequest
from pinecone.grpc.utils import dict_to_proto_struct


class TestGrpcIndexUpdate:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo.pinecone.io")
        self.index = GRPCIndex(
            config=self.config, index_name="example-name", _endpoint_override="test-endpoint"
        )

    def test_update_byIdAnValues_updateByIdAndValues(self, mocker, vals1):
        mocker.patch.object(self.index.runner, "run", autospec=True)
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
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.update("vec1", values=vals1, set_metadata=md1)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(id="vec1", values=vals1, set_metadata=dict_to_proto_struct(md1)),
            timeout=None,
        )

    def test_update_byFilter_updateByFilter(self, mocker, md1):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        filter_dict = {"genre": {"$eq": "comedy"}}
        self.index.update(filter=filter_dict, set_metadata=md1, namespace="ns")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(
                filter=dict_to_proto_struct(filter_dict),
                set_metadata=dict_to_proto_struct(md1),
                namespace="ns",
            ),
            timeout=None,
        )

    def test_update_byFilterWithDryRun_updateByFilterWithDryRun(self, mocker, md1):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        filter_dict = {"year": {"$gte": 2020}}
        self.index.update(filter=filter_dict, set_metadata=md1, dry_run=True, namespace="ns")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(
                filter=dict_to_proto_struct(filter_dict),
                set_metadata=dict_to_proto_struct(md1),
                dry_run=True,
                namespace="ns",
            ),
            timeout=None,
        )

    def test_update_byFilterAsync_updateByFilterAsync(self, mocker, md1):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        filter_dict = {"status": "active"}
        self.index.update(filter=filter_dict, set_metadata=md1, async_req=True, namespace="ns")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update.future,
            UpdateRequest(
                filter=dict_to_proto_struct(filter_dict),
                set_metadata=dict_to_proto_struct(md1),
                namespace="ns",
            ),
            timeout=None,
        )

    def test_update_bothIdAndFilter_raisesValueError(self, mocker):
        import pytest

        with pytest.raises(ValueError, match="Cannot provide both 'id' and 'filter'"):
            self.index.update(id="vec1", filter={"genre": "comedy"})

    def test_update_neitherIdNorFilter_raisesValueError(self, mocker):
        import pytest

        with pytest.raises(ValueError, match="Either 'id' or 'filter' must be provided"):
            self.index.update(values=[0.1, 0.2, 0.3])

    def test_update_byFilter_returnsMatchedRecords(self, mocker, md1):
        filter_dict = {"genre": {"$eq": "comedy"}}
        # Create a mock response dict that parse_update_response will convert
        response_dict = {"matchedRecords": 5}
        mocker.patch.object(self.index.runner, "run", return_value=response_dict)

        result = self.index.update(filter=filter_dict, set_metadata=md1, namespace="ns")
        assert result["matched_records"] == 5
