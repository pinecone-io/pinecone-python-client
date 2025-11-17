from typing import TypedDict, Any, cast

from pinecone.utils import parse_non_empty_args
from pinecone.core.openapi.db_data.model.create_namespace_request import CreateNamespaceRequest
from pinecone.core.openapi.db_data.model.create_namespace_request_schema import (
    CreateNamespaceRequestSchema,
)


class DescribeNamespaceArgs(TypedDict, total=False):
    namespace: str


class DeleteNamespaceArgs(TypedDict, total=False):
    namespace: str


class CreateNamespaceArgs(TypedDict, total=False):
    create_namespace_request: CreateNamespaceRequest


class NamespaceRequestFactory:
    @staticmethod
    def describe_namespace_args(namespace: str, **kwargs) -> DescribeNamespaceArgs:
        if not isinstance(namespace, str):
            raise ValueError("namespace must be string")
        base_args = {"namespace": namespace}
        return cast(DescribeNamespaceArgs, {**base_args, **kwargs})

    @staticmethod
    def delete_namespace_args(namespace: str, **kwargs) -> DeleteNamespaceArgs:
        if not isinstance(namespace, str):
            raise ValueError("namespace must be string")
        base_args = {"namespace": namespace}
        return cast(DeleteNamespaceArgs, {**base_args, **kwargs})

    @staticmethod
    def create_namespace_args(
        name: str, schema: (CreateNamespaceRequestSchema | dict[str, Any]) | None = None, **kwargs
    ) -> CreateNamespaceArgs:
        if not isinstance(name, str):
            raise ValueError("name must be string")
        if name.strip() == "":
            raise ValueError("name must not be empty")

        request_kwargs: dict[str, Any] = {"name": name}
        if schema is not None:
            if isinstance(schema, dict):
                schema_obj = CreateNamespaceRequestSchema(**schema)
                request_kwargs["schema"] = schema_obj
            else:
                # schema is already CreateNamespaceRequestSchema
                request_kwargs["schema"] = schema

        create_namespace_request = CreateNamespaceRequest(**request_kwargs)
        base_args = {"create_namespace_request": create_namespace_request}
        return cast(CreateNamespaceArgs, {**base_args, **kwargs})

    @staticmethod
    def list_namespaces_args(
        limit: int | None = None, pagination_token: str | None = None, **kwargs
    ) -> dict[str, Any]:
        base_args = parse_non_empty_args([("limit", limit), ("pagination_token", pagination_token)])
        return {**base_args, **kwargs}
