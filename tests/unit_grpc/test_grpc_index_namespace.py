from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
    DescribeNamespaceRequest,
    DeleteNamespaceRequest,
    ListNamespacesRequest,
)


class TestGrpcIndexNamespace:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo.pinecone.io")
        self.index = GRPCIndex(
            config=self.config, index_name="example-name", _endpoint_override="test-endpoint"
        )

    def test_describe_namespace(self, mocker):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.describe_namespace(namespace="test_namespace")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.DescribeNamespace,
            DescribeNamespaceRequest(namespace="test_namespace"),
            timeout=None,
        )

    def test_describe_namespace_with_timeout(self, mocker):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.describe_namespace(namespace="test_namespace", timeout=30)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.DescribeNamespace,
            DescribeNamespaceRequest(namespace="test_namespace"),
            timeout=30,
        )

    def test_delete_namespace(self, mocker):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.delete_namespace(namespace="test_namespace")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.DeleteNamespace,
            DeleteNamespaceRequest(namespace="test_namespace"),
            timeout=None,
        )

    def test_delete_namespace_with_timeout(self, mocker):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.delete_namespace(namespace="test_namespace", timeout=30)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.DeleteNamespace,
            DeleteNamespaceRequest(namespace="test_namespace"),
            timeout=30,
        )

    def test_list_namespaces_paginated(self, mocker):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.list_namespaces_paginated(limit=10, pagination_token="token123")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.ListNamespaces,
            ListNamespacesRequest(limit=10, pagination_token="token123"),
            timeout=None,
        )

    def test_list_namespaces_paginated_with_timeout(self, mocker):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.list_namespaces_paginated(limit=10, timeout=30)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.ListNamespaces, ListNamespacesRequest(limit=10), timeout=30
        )

    def test_list_namespaces_paginated_no_args(self, mocker):
        mocker.patch.object(self.index.runner, "run", autospec=True)
        self.index.list_namespaces_paginated()
        self.index.runner.run.assert_called_once_with(
            self.index.stub.ListNamespaces, ListNamespacesRequest(), timeout=None
        )
