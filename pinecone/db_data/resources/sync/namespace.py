from typing import Iterator, Any

from pinecone.core.openapi.db_data.api.namespace_operations_api import NamespaceOperationsApi
from pinecone.core.openapi.db_data.models import ListNamespacesResponse, NamespaceDescription

from pinecone.utils import install_json_repr_override, PluginAware, require_kwargs

from .namespace_request_factory import NamespaceRequestFactory

for m in [ListNamespacesResponse, NamespaceDescription]:
    install_json_repr_override(m)


class NamespaceResource(PluginAware):
    def __init__(self, api_client, config, openapi_config, pool_threads: int) -> None:
        self.config = config
        """ :meta private: """

        self._openapi_config = openapi_config
        """ :meta private: """

        self._pool_threads = pool_threads
        """ :meta private: """

        self.__namespace_operations_api = NamespaceOperationsApi(api_client)
        super().__init__()

    @require_kwargs
    def create(self, name: str, schema: Any | None = None, **kwargs) -> NamespaceDescription:
        """
        Args:
            name (str): The name of the namespace to create
            schema (Optional[Any]): Optional schema configuration for the namespace. Can be a dictionary or CreateNamespaceRequestSchema object. [optional]

        Returns:
            ``NamespaceDescription``: Information about the created namespace including vector count

        Create a namespace in a serverless index. For guidance and examples, see
        `Manage namespaces <https://docs.pinecone.io/guides/manage-data/manage-namespaces>`_.

        **Note:** This operation is not supported for pod-based indexes.
        """
        from typing import cast

        args = NamespaceRequestFactory.create_namespace_args(name=name, schema=schema, **kwargs)
        result = self.__namespace_operations_api.create_namespace(**args)
        return cast(NamespaceDescription, result)

    @require_kwargs
    def describe(self, namespace: str, **kwargs) -> NamespaceDescription:
        """
        Args:
            namespace (str): The namespace to describe

        Returns:
            ``NamespaceDescription``: Information about the namespace including vector count

        Describe a namespace within an index, showing the vector count within the namespace.
        """
        from typing import cast

        args = NamespaceRequestFactory.describe_namespace_args(namespace=namespace, **kwargs)
        result = self.__namespace_operations_api.describe_namespace(**args)
        return cast(NamespaceDescription, result)

    @require_kwargs
    def delete(self, namespace: str, **kwargs):
        """
        Args:
            namespace (str): The namespace to delete

        Delete a namespace from an index.
        """
        args = NamespaceRequestFactory.delete_namespace_args(namespace=namespace, **kwargs)
        return self.__namespace_operations_api.delete_namespace(**args)

    @require_kwargs
    def list(self, limit: int | None = None, **kwargs) -> Iterator[ListNamespacesResponse]:
        """
        Args:
            limit (Optional[int]): The maximum number of namespaces to fetch in each network call. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): When there are multiple pages of results, a pagination token is returned in the response. The token can be used
                to fetch the next page of results. [optional]

        Returns:
            Returns a generator that yields each namespace. It automatically handles pagination tokens on your behalf so you can
            easily iterate over all results. The ``list`` method accepts all of the same arguments as list_paginated

        .. code-block:: python
            for namespace in index.list_namespaces():
                print(namespace)

        You can convert the generator into a list by wrapping the generator in a call to the built-in ``list`` function:

        .. code-block:: python
            namespaces = list(index.list_namespaces())

        You should be cautious with this approach because it will fetch all namespaces at once, which could be a large number
        of network calls and a lot of memory to hold the results.
        """
        done = False
        while not done:
            results = self.list_paginated(limit=limit, **kwargs)
            if results.namespaces is not None and len(results.namespaces) > 0:
                for namespace in results.namespaces:
                    yield namespace

            if results.pagination and results.pagination.next:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

    @require_kwargs
    def list_paginated(
        self, limit: int | None = None, pagination_token: str | None = None, **kwargs
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
                >>> results = index.list_paginated(limit=5)
                >>> results.pagination.next
                eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
                >>> next_results = index.list_paginated(limit=5, pagination_token=results.pagination.next)
        """
        from typing import cast

        args = NamespaceRequestFactory.list_namespaces_args(
            limit=limit, pagination_token=pagination_token, **kwargs
        )
        result = self.__namespace_operations_api.list_namespaces_operation(**args)
        return cast(ListNamespacesResponse, result)
