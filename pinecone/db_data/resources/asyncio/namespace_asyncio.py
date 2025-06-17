from typing import Optional, AsyncIterator

from pinecone.core.openapi.db_data.api.namespace_operations_api import AsyncioNamespaceOperationsApi
from pinecone.core.openapi.db_data.models import (
    ListNamespacesResponse,
    NamespaceDescription,
)

from pinecone.utils import install_json_repr_override, require_kwargs

from ..sync.namespace_request_factory import NamespaceRequestFactory

for m in [ListNamespacesResponse, NamespaceDescription]:
    install_json_repr_override(m)


class NamespaceResourceAsyncio:
    def __init__(self, api_client) -> None:
        self.__namespace_operations_api = AsyncioNamespaceOperationsApi(api_client)

    @require_kwargs
    async def describe(self, namespace: str, **kwargs) -> NamespaceDescription:
        """
        Args:
            namespace (str): The namespace to describe

        Returns:
            ``NamespaceDescription``: Information about the namespace including vector count

        Describe a namespace within an index, showing the vector count within the namespace.
        """
        args = NamespaceRequestFactory.describe_namespace_args(namespace=namespace, **kwargs)
        return await self.__namespace_operations_api.describe_namespace(**args)

    @require_kwargs
    async def delete(self, namespace: str, **kwargs):
        """
        Args:
            namespace (str): The namespace to delete

        Delete a namespace from an index.
        """
        args = NamespaceRequestFactory.delete_namespace_args(namespace=namespace, **kwargs)
        return await self.__namespace_operations_api.delete_namespace(**args)

    @require_kwargs
    async def list(self, limit: Optional[int] = None, **kwargs) -> AsyncIterator[ListNamespacesResponse]:
        """
        Args:
            limit (Optional[int]): The maximum number of namespaces to fetch in each network call. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): When there are multiple pages of results, a pagination token is returned in the response. The token can be used
                to fetch the next page of results. [optional]

        Returns:
            Returns an async generator that yields each namespace. It automatically handles pagination tokens on your behalf so you can
            easily iterate over all results. The ``list`` method accepts all of the same arguments as list_paginated

        .. code-block:: python
            async for namespace in index.list_namespaces():
                print(namespace)

        You can convert the generator into a list by using an async list comprehension:

        .. code-block:: python
            namespaces = [namespace async for namespace in index.list_namespaces()]

        You should be cautious with this approach because it will fetch all namespaces at once, which could be a large number
        of network calls and a lot of memory to hold the results.
        """
        done = False
        while not done:
            results = await self.list_paginated(limit=limit, **kwargs)
            if results.namespaces is not None and len(results.namespaces) > 0:
                for namespace in results.namespaces:
                    yield namespace

            if results.pagination and results.pagination.next:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

    @require_kwargs
    async def list_paginated(
        self, limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> ListNamespacesResponse:
        """
        Args:
            limit (Optional[int]): The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]

        Returns:
            ``ListNamespacesResponse``: Object containing the list of namespaces and pagination information.

        List all namespaces in an index with pagination support. The response includes pagination information if there are more results available.

        Consider using the ``list`` method to avoid having to handle pagination tokens manually.

        Examples:
            .. code-block:: python
                >>> results = await index.list_paginated(limit=5)
                >>> results.pagination.next
                eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
                >>> next_results = await index.list_paginated(limit=5, pagination_token=results.pagination.next)
        """
        args = NamespaceRequestFactory.list_namespaces_args(limit=limit, pagination_token=pagination_token, **kwargs)
        return await self.__namespace_operations_api.list_namespaces_operation(**args)