from typing import Optional, TypedDict, Any

from pinecone.utils import parse_non_empty_args


class DescribeNamespaceArgs(TypedDict, total=False):
    namespace: str


class DeleteNamespaceArgs(TypedDict, total=False):
    namespace: str


class NamespaceRequestFactory:
    @staticmethod
    def describe_namespace_args(namespace: str) -> DescribeNamespaceArgs:
        return {"namespace": namespace}

    @staticmethod
    def delete_namespace_args(namespace: str) -> DeleteNamespaceArgs:
        if isinstance(namespace, int):
            namespace = str(namespace)
        return {"namespace": namespace}

    @staticmethod
    def list_namespaces_args(
        limit: Optional[int] = None, pagination_token: Optional[str] = None
    ) -> dict[str, Any]:
        return parse_non_empty_args([("limit", limit), ("pagination_token", pagination_token)]) 