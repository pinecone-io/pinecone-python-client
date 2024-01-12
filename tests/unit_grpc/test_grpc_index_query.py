from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone.core.grpc.protos.vector_service_pb2 import (
    QueryRequest,
    QueryVector,
)
from pinecone.grpc.utils import dict_to_proto_struct


class TestGrpcIndexQuery:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo")
        self.index = GRPCIndex(config=self.config, index_name="example-name", _endpoint_override="test-endpoint")

    def test_query_byVectorNoFilter_queryVectorNoFilter(self, mocker, vals1):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.query(top_k=10, vector=vals1)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(top_k=10, vector=vals1),
            timeout=None,
        )

    def test_query_byVectorWithFilter_queryVectorWithFilter(self, mocker, vals1, filter1):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.query(top_k=10, vector=vals1, filter=filter1, namespace="ns", timeout=10)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(top_k=10, vector=vals1, filter=dict_to_proto_struct(filter1), namespace="ns"),
            timeout=10,
        )

    def test_query_byTuplesNoFilter_queryVectorsNoFilter(self, mocker, vals1, vals2):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.query(top_k=10, queries=[(vals1,), (vals2,)])
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(
                queries=[QueryVector(values=vals1, filter={}), QueryVector(values=vals2, filter={})], top_k=10
            ),
            timeout=None,
        )

    def test_query_byTuplesWithFilter_queryVectorsWithFilter(self, mocker, vals1, vals2, filter1, filter2):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.query(top_k=10, queries=[(vals1, filter1), (vals2, filter2)])
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(
                queries=[
                    QueryVector(values=vals1, filter=dict_to_proto_struct(filter1)),
                    QueryVector(values=vals2, filter=dict_to_proto_struct(filter2)),
                ],
                top_k=10,
            ),
            timeout=None,
        )

    def test_query_byVecId_queryByVecId(self, mocker):
        mocker.patch.object(self.index, "_wrap_grpc_call", autospec=True)
        self.index.query(top_k=10, id="vec1", include_metadata=True, include_values=False)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(top_k=10, id="vec1", include_metadata=True, include_values=False),
            timeout=None,
        )
