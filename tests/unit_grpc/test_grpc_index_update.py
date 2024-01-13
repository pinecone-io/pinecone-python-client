from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone.core.grpc.protos.vector_service_pb2 import (
    UpdateRequest,
)
from pinecone.grpc.utils import dict_to_proto_struct


class TestGrpcIndexUpdate:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo")
        self.index = GRPCIndex(config=self.config, index_name="example-name", _endpoint_override="test-endpoint")

    def test_update_byIdAnValues_updateByIdAndValues(self, mocker, vals1):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.update(id="vec1", values=vals1, namespace="ns", timeout=30)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(id="vec1", values=vals1, namespace="ns"),
            timeout=30,
        )

    def test_update_byIdAnValuesAsync_updateByIdAndValuesAsync(self, mocker, vals1):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.update(id="vec1", values=vals1, namespace="ns", timeout=30, async_req=True)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Update.future,
            UpdateRequest(id="vec1", values=vals1, namespace="ns"),
            timeout=30,
        )

    def test_update_byIdAnValuesAndMetadata_updateByIdAndValuesAndMetadata(self, mocker, vals1, md1):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.update("vec1", values=vals1, set_metadata=md1)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(id="vec1", values=vals1, set_metadata=dict_to_proto_struct(md1)),
            timeout=None,
        )
