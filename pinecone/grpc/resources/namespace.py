import logging
from typing import Optional, Dict, Any, Union

from pinecone.utils import require_kwargs
from ..utils import (
    dict_to_proto_struct,
    parse_namespace_description,
    parse_list_namespaces_response,
    parse_delete_response,
)
from pinecone.core.openapi.db_data.models import (
    NamespaceDescription,
    ListNamespacesResponse,
)
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
    DescribeNamespaceRequest,
    DeleteNamespaceRequest,
    ListNamespacesRequest,
    CreateNamespaceRequest,
    MetadataSchema,
    MetadataFieldProperties,
)
from ..future import PineconeGrpcFuture

logger = logging.getLogger(__name__)
""" :meta private: """


class NamespaceResourceGRPC:
    """Resource for namespace operations on a Pinecone index via GRPC."""

    def __init__(self, stub, runner):
        self.stub = stub
        """ :meta private: """
        self.runner = runner
        """ :meta private: """

    @staticmethod
    def _parse_non_empty_args(args: list) -> Dict[str, Any]:
        return {arg_name: val for arg_name, val in args if val is not None}

    @require_kwargs
    def create(
        self, name: str, schema: Optional[Dict[str, Any]] = None, async_req: bool = False, **kwargs
    ) -> Union[NamespaceDescription, PineconeGrpcFuture]:
        timeout = kwargs.pop("timeout", None)

        # Build MetadataSchema from dict if provided
        metadata_schema = None
        if schema is not None:
            if isinstance(schema, dict):
                # Convert dict to MetadataSchema
                fields = {}
                for key, value in schema.get("fields", {}).items():
                    if isinstance(value, dict):
                        filterable = value.get("filterable", False)
                        fields[key] = MetadataFieldProperties(filterable=filterable)
                    else:
                        # If value is already a MetadataFieldProperties, use it directly
                        fields[key] = value
                metadata_schema = MetadataSchema(fields=fields)
            else:
                # Assume it's already a MetadataSchema
                metadata_schema = schema

        request_kwargs: Dict[str, Any] = {"name": name}
        if metadata_schema is not None:
            request_kwargs["schema"] = metadata_schema

        request = CreateNamespaceRequest(**request_kwargs)

        if async_req:
            future = self.runner.run(self.stub.CreateNamespace.future, request, timeout=timeout)
            return PineconeGrpcFuture(
                future, timeout=timeout, result_transformer=parse_namespace_description
            )

        response = self.runner.run(self.stub.CreateNamespace, request, timeout=timeout)
        return parse_namespace_description(response)

    @require_kwargs
    def describe(self, namespace: str, **kwargs) -> NamespaceDescription:
        timeout = kwargs.pop("timeout", None)
        request = DescribeNamespaceRequest(namespace=namespace)
        response = self.runner.run(self.stub.DescribeNamespace, request, timeout=timeout)
        return parse_namespace_description(response)

    @require_kwargs
    def delete(self, namespace: str, **kwargs) -> Dict[str, Any]:
        timeout = kwargs.pop("timeout", None)
        request = DeleteNamespaceRequest(namespace=namespace)
        response = self.runner.run(self.stub.DeleteNamespace, request, timeout=timeout)
        return parse_delete_response(response)

    @require_kwargs
    def list_paginated(
        self, limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> ListNamespacesResponse:
        args_dict = self._parse_non_empty_args(
            [("limit", limit), ("pagination_token", pagination_token)]
        )
        timeout = kwargs.pop("timeout", None)
        request = ListNamespacesRequest(**args_dict, **kwargs)
        response = self.runner.run(self.stub.ListNamespaces, request, timeout=timeout)
        return parse_list_namespaces_response(response)

    @require_kwargs
    def list(self, limit: Optional[int] = None, **kwargs):
        done = False
        while not done:
            try:
                results = self.list_paginated(limit=limit, **kwargs)
            except Exception as e:
                raise e

            if results.namespaces and len(results.namespaces) > 0:
                for namespace in results.namespaces:
                    yield namespace

            if results.pagination and results.pagination.next:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

