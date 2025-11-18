import pytest
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

    def test_update_withFilter_updateWithFilter(self, mocker, filter1):
        mock_response = UpdateResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.update(filter=filter1, namespace="ns")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(filter=dict_to_proto_struct(filter1), namespace="ns"),
            timeout=None,
        )

    def test_update_withFilterAndSetMetadata_updateWithFilterAndSetMetadata(
        self, mocker, md1, filter1
    ):
        mock_response = UpdateResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.update(set_metadata=md1, filter=filter1, namespace="ns")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(
                set_metadata=dict_to_proto_struct(md1),
                filter=dict_to_proto_struct(filter1),
                namespace="ns",
            ),
            timeout=None,
        )

    def test_update_withFilterAndValues_updateWithFilterAndValues(self, mocker, vals1, filter1):
        mock_response = UpdateResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.update(values=vals1, filter=filter1, namespace="ns")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(values=vals1, filter=dict_to_proto_struct(filter1), namespace="ns"),
            timeout=None,
        )

    def test_update_withFilter_asyncReq_updateWithFilterAsyncReq(self, mocker, filter1):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.update(filter=filter1, namespace="ns", async_req=True)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update.future,
            UpdateRequest(filter=dict_to_proto_struct(filter1), namespace="ns"),
            timeout=None,
        )

    def test_update_withFilterOnly_noId(self, mocker, filter1, md1):
        """Test update with filter only (no id) for bulk updates."""
        mock_response = UpdateResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.update(set_metadata=md1, filter=filter1, namespace="ns")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(
                set_metadata=dict_to_proto_struct(md1),
                filter=dict_to_proto_struct(filter1),
                namespace="ns",
            ),
            timeout=None,
        )

    def test_update_withNeitherIdNorFilter_raisesError(self, mocker, vals1):
        """Test that update raises error when neither id nor filter is provided."""
        mocker.patch.object(self.index.runner, "run", autospec=True)
        with pytest.raises(ValueError, match="Either 'id' or 'filter' must be provided"):
            self.index.update(values=vals1, namespace="ns")

    def test_update_withBothIdAndFilter_raisesError(self, mocker, vals1, filter1):
        """Test that update raises error when both id and filter are provided."""
        mocker.patch.object(self.index.runner, "run", autospec=True)
        with pytest.raises(ValueError, match="Cannot provide both 'id' and 'filter'"):
            self.index.update(id="vec1", filter=filter1, values=vals1, namespace="ns")

    def test_update_withDryRun_updateWithDryRun(self, mocker, filter1):
        """Test update with dry_run parameter."""
        mock_response = UpdateResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.update(filter=filter1, dry_run=True, namespace="ns")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(filter=dict_to_proto_struct(filter1), dry_run=True, namespace="ns"),
            timeout=None,
        )

    def test_update_withDryRunAndSetMetadata_updateWithDryRunAndSetMetadata(
        self, mocker, md1, filter1
    ):
        """Test update with dry_run and set_metadata."""
        mock_response = UpdateResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.update(set_metadata=md1, filter=filter1, dry_run=True, namespace="ns")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(
                set_metadata=dict_to_proto_struct(md1),
                filter=dict_to_proto_struct(filter1),
                dry_run=True,
                namespace="ns",
            ),
            timeout=None,
        )

    def test_update_withDryRunFalse_updateWithDryRunFalse(self, mocker, filter1):
        """Test update with dry_run=False."""
        mock_response = UpdateResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.update(filter=filter1, dry_run=False, namespace="ns")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(filter=dict_to_proto_struct(filter1), dry_run=False, namespace="ns"),
            timeout=None,
        )

    def test_update_withDryRun_asyncReq_updateWithDryRunAsyncReq(self, mocker, filter1):
        """Test update with dry_run and async_req=True."""
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.update(filter=filter1, dry_run=True, namespace="ns", async_req=True)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.Update.future,
            UpdateRequest(filter=dict_to_proto_struct(filter1), dry_run=True, namespace="ns"),
            timeout=None,
        )
