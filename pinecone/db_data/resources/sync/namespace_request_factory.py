from typing import Optional, TypedDict, Any, cast

from pinecone.utils import parse_non_empty_args


class DescribeNamespaceArgs(TypedDict, total=False):
    namespace: str


class DeleteNamespaceArgs(TypedDict, total=False):
    namespace: str


class NamespaceRequestFactory:
    @staticmethod
    def describe_namespace_args(namespace: str, **kwargs) -> DescribeNamespaceArgs:
        if not isinstance(namespace, str):
            raise ValueError('namespace must be string')
        base_args = {"namespace": namespace}
        return cast(DescribeNamespaceArgs, {**base_args, **kwargs})

    @staticmethod
    def delete_namespace_args(namespace: str, **kwargs) -> DeleteNamespaceArgs:
        if not isinstance(namespace, str):
            raise ValueError('namespace must be string')
        base_args = {"namespace": namespace}
        return cast(DeleteNamespaceArgs, {**base_args, **kwargs})

    @staticmethod
    def list_namespaces_args(
        limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> dict[str, Any]:
        base_args = parse_non_empty_args([("limit", limit), ("pagination_token", pagination_token)])
        return {**base_args, **kwargs}