from pinecone import Config
from pinecone.grpc import GRPCIndex
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
    CreateNamespaceRequest,
    DescribeNamespaceRequest,
    DeleteNamespaceRequest,
    ListNamespacesRequest,
    MetadataSchema,
    NamespaceDescription as GRPCNamespaceDescription,
    ListNamespacesResponse as GRPCListNamespacesResponse,
)


class TestGrpcIndexNamespace:
    def setup_method(self):
        self.config = Config(api_key="test-api-key", host="foo.pinecone.io")
        self.index = GRPCIndex(
            config=self.config, index_name="example-name", _endpoint_override="test-endpoint"
        )

    def test_create_namespace(self, mocker):
        mock_response = GRPCNamespaceDescription()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.create_namespace(name="test_namespace")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.CreateNamespace,
            CreateNamespaceRequest(name="test_namespace"),
            timeout=None,
        )

    def test_create_namespace_with_timeout(self, mocker):
        mock_response = GRPCNamespaceDescription()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.create_namespace(name="test_namespace", timeout=30)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.CreateNamespace,
            CreateNamespaceRequest(name="test_namespace"),
            timeout=30,
        )

    def test_create_namespace_with_schema(self, mocker):
        mock_response = GRPCNamespaceDescription()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        schema_dict = {"fields": {"field1": {"filterable": True}, "field2": {"filterable": False}}}
        self.index.create_namespace(name="test_namespace", schema=schema_dict)
        call_args = self.index.runner.run.call_args
        assert call_args[0][0] == self.index.stub.CreateNamespace
        request = call_args[0][1]
        assert isinstance(request, CreateNamespaceRequest)
        assert request.name == "test_namespace"
        assert isinstance(request.schema, MetadataSchema)
        assert "field1" in request.schema.fields
        assert "field2" in request.schema.fields
        assert request.schema.fields["field1"].filterable is True
        assert request.schema.fields["field2"].filterable is False

    def test_describe_namespace(self, mocker):
        mock_response = GRPCNamespaceDescription()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.describe_namespace(namespace="test_namespace")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.DescribeNamespace,
            DescribeNamespaceRequest(namespace="test_namespace"),
            timeout=None,
        )

    def test_describe_namespace_with_timeout(self, mocker):
        mock_response = GRPCNamespaceDescription()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.describe_namespace(namespace="test_namespace", timeout=30)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.DescribeNamespace,
            DescribeNamespaceRequest(namespace="test_namespace"),
            timeout=30,
        )

    def test_delete_namespace(self, mocker):
        mock_response = mocker.Mock()  # DeleteResponse is just a dict
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.delete_namespace(namespace="test_namespace")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.DeleteNamespace,
            DeleteNamespaceRequest(namespace="test_namespace"),
            timeout=None,
        )

    def test_delete_namespace_with_timeout(self, mocker):
        mock_response = mocker.Mock()  # DeleteResponse is just a dict
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.delete_namespace(namespace="test_namespace", timeout=30)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.DeleteNamespace,
            DeleteNamespaceRequest(namespace="test_namespace"),
            timeout=30,
        )

    def test_list_namespaces_paginated(self, mocker):
        mock_response = GRPCListNamespacesResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.list_namespaces_paginated(limit=10, pagination_token="token123")
        self.index.runner.run.assert_called_once_with(
            self.index.stub.ListNamespaces,
            ListNamespacesRequest(limit=10, pagination_token="token123"),
            timeout=None,
        )

    def test_list_namespaces_paginated_with_timeout(self, mocker):
        mock_response = GRPCListNamespacesResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.list_namespaces_paginated(limit=10, timeout=30)
        self.index.runner.run.assert_called_once_with(
            self.index.stub.ListNamespaces, ListNamespacesRequest(limit=10), timeout=30
        )

    def test_list_namespaces_paginated_no_args(self, mocker):
        mock_response = GRPCListNamespacesResponse()
        mocker.patch.object(self.index.runner, "run", return_value=(mock_response, None))
        self.index.list_namespaces_paginated()
        self.index.runner.run.assert_called_once_with(
            self.index.stub.ListNamespaces, ListNamespacesRequest(), timeout=None
        )
