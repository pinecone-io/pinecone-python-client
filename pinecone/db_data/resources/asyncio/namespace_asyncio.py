from typing import Optional, AsyncIterator, Union

from pinecone.core.openapi.db_data.api.namespace_operations_api import AsyncioNamespaceOperationsApi
from pinecone.core.openapi.db_data.models import (
    ListNamespacesResponse,
    NamespaceDescription,
)

from pinecone.utils import install_json_repr_override

from ..sync.namespace_request_factory import NamespaceRequestFactory

for m in [ListNamespacesResponse, NamespaceDescription]:
    install_json_repr_override(m)


class NamespaceResourceAsyncio:
    def __init__(self, api_client, **kwargs) -> None:
        self.__namespace_operations_api = AsyncioNamespaceOperationsApi(api_client)

    async def describe(self, namespace: str) -> NamespaceDescription:
        """
        Args:
            namespace (str): The namespace to describe

        Returns:
            `NamespaceDescription`: Information about the namespace including vector count

        Describe a namespace within an index, showing the vector count within the namespace.
        """
        args = NamespaceRequestFactory.describe_namespace_args(namespace=namespace)
        return await self.__namespace_operations_api.describe_namespace(**args)

    async def delete(self, namespace: str):
        """
        Args:
            namespace (str): The namespace to delete

        Delete a namespace from an index.
        """
        args = NamespaceRequestFactory.delete_namespace_args(namespace=namespace)
        return await self.__namespace_operations_api.delete_namespace(**args)

    async def list(self, **kwargs) -> AsyncIterator[NamespaceDescription]:
        """
        Args:
            limit (Optional[int]): The maximum number of namespaces to fetch in each network call. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): When there are multiple pages of results, a pagination token is returned in the response. The token can be used
                to fetch the next page of results. [optional]

        Returns:
            Returns an async generator that yields each namespace. It automatically handles pagination tokens on your behalf so you can
            easily iterate over all results.

        ```python
        async for namespace in index.list_namespaces():
            print(namespace)
        ```
        """
        done = False
        while not done:
            args_dict = NamespaceRequestFactory.list_namespaces_args(**kwargs)
            results = await self.__namespace_operations_api.list_namespaces_operation(**args_dict)
            if len(results.namespaces) > 0:
                for namespace in results.namespaces:
                    yield namespace

            if results.pagination:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True 