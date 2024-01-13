from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone import DescribeIndexStatsRequest
from pinecone.core.grpc.protos.vector_service_pb2 import (
    DescribeIndexStatsRequest,
)
from pinecone.grpc.utils import dict_to_proto_struct


class TestGrpcIndexDescribeIndexStats:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo")
        self.index = GRPCIndex(config=self.config, index_name="example-name", _endpoint_override="test-endpoint")

    def test_describeIndexStats_callWithoutFilter_CalledWithoutFilter(self, mocker):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.describe_index_stats()
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.DescribeIndexStats,
            DescribeIndexStatsRequest(),
            timeout=None,
        )

    def test_describeIndexStats_callWithFilter_CalledWithFilter(self, mocker, filter1):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.describe_index_stats(filter=filter1)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.DescribeIndexStats,
            DescribeIndexStatsRequest(filter=dict_to_proto_struct(filter1)),
            timeout=None,
        )
