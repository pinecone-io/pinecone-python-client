from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
    DescribeIndexStatsRequest,
    DescribeIndexStatsResponse,
)
from pinecone.grpc.utils import dict_to_proto_struct
from google.protobuf import json_format


class TestGrpcIndexDescribeIndexStats:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo.pinecone.io")
        self.index = GRPCIndex(
            config=self.config, index_name="example-name", _endpoint_override="test-endpoint"
        )

    def test_describeIndexStats_callWithoutFilter_CalledWithoutFilter(self, mocker):
        # Create a proper protobuf response
        response = DescribeIndexStatsResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(response, None))
        mocker.patch.object(json_format, "MessageToDict", return_value={})
        mocker.patch("pinecone.grpc.index_grpc.parse_stats_response", return_value={})
        self.index.describe_index_stats()
        self.index.runner.run.assert_called_once_with(
            self.index.stub.DescribeIndexStats, DescribeIndexStatsRequest(), timeout=None
        )

    def test_describeIndexStats_callWithFilter_CalledWithFilter(self, mocker, filter1):
        # Create a proper protobuf response
        response = DescribeIndexStatsResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(response, None))
        mocker.patch.object(json_format, "MessageToDict", return_value={})
        mocker.patch("pinecone.grpc.index_grpc.parse_stats_response", return_value={})
        self.index.describe_index_stats(filter=filter1)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.DescribeIndexStats,
            DescribeIndexStatsRequest(filter=dict_to_proto_struct(filter1)),
            timeout=None,
        )
