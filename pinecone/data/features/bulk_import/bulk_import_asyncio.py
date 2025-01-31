from typing import Optional, Literal, AsyncIterator, List

from pinecone.core.openapi.db_data.api.bulk_operations_api import AsyncioBulkOperationsApi

from pinecone.utils import install_json_repr_override

from pinecone.core.openapi.db_data.models import (
    StartImportResponse,
    ListImportsResponse,
    ImportModel,
)

from .bulk_import_request_factory import BulkImportRequestFactory

for m in [StartImportResponse, ListImportsResponse, ImportModel]:
    install_json_repr_override(m)


class ImportFeatureMixinAsyncio:
    def __init__(self, api_client, **kwargs):
        self.__import_operations_api = AsyncioBulkOperationsApi(api_client)

    async def start_import(
        self,
        uri: str,
        integration_id: Optional[str] = None,
        error_mode: Optional[Literal["CONTINUE", "ABORT"]] = "CONTINUE",
    ) -> StartImportResponse:
        """Import data from a storage provider into an index. The uri must start with the scheme of a supported
        storage provider. For buckets that are not publicly readable, you will also need to separately configure
        a storage integration and pass the integration id.

        Examples:
            >>> from pinecone import Pinecone
            >>> index = Pinecone().AsyncioIndex(host="example-index.svc.aped-4627-b74a.pinecone.io")
            >>> await index.start_import(uri="s3://bucket-name/path/to/data.parquet")
            { id: "1" }

        Args:
            uri (str): The URI of the data to import. The URI must start with the scheme of a supported storage provider.
            integration_id (Optional[str], optional): If your bucket requires authentication to access, you need to pass the id of your storage integration using this property. Defaults to None.
            error_mode: Defaults to "CONTINUE". If set to "CONTINUE", the import operation will continue even if some
                records fail to import. Pass "ABORT" to stop the import operation if any records fail to import.

        Returns:
            StartImportResponse: Contains the id of the import operation.
        """
        req = BulkImportRequestFactory.start_import_request(
            uri=uri, integration_id=integration_id, error_mode=error_mode
        )
        return await self.__import_operations_api.start_bulk_import(req)

    async def list_imports(self, **kwargs) -> AsyncIterator[List[ImportModel]]:
        """
        Returns an async generator that yields each import operation. It automatically handles pagination tokens on your behalf so you can
        easily iterate over all results. The `list_imports` method accepts all of the same arguments as list_imports_paginated

        ```python
        async for op in index.list_imports():
            print(op)
        ```

        Args:
            limit (Optional[int]): The maximum number of operations to fetch in each network call. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): When there are multiple pages of results, a pagination token is returned in the response. The token can be used
                to fetch the next page of results. [optional]
        """
        done = False
        while not done:
            results = await self.list_imports_paginated(**kwargs)
            if len(results.data) > 0:
                for op in results.data:
                    yield op

            if results.pagination:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

    async def list_imports_paginated(
        self, limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> ListImportsResponse:
        """
        The list_imports_paginated operation returns information about import operations.
        It returns operations in a paginated form, with a pagination token to fetch the next page of results.

        Consider using the `list_imports` method to avoid having to handle pagination tokens manually.

        Examples:
            >>> results = await index.list_imports_paginated(limit=5)
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> results.data[0]
            {
                "id": "6",
                "uri": "s3://dev-bulk-import-datasets-pub/10-records-dim-10/",
                "status": "Completed",
                "percent_complete": 100.0,
                "records_imported": 10,
                "created_at": "2024-09-06T14:52:02.567776+00:00",
                "finished_at": "2024-09-06T14:52:28.130717+00:00"
            }
            >>> next_results = await index.list_imports_paginated(limit=5, pagination_token=results.pagination.next)

        Args:
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]

        Returns: ListImportsResponse object which contains the list of operations as ImportModel objects, pagination information,
            and usage showing the number of read_units consumed.
        """
        args_dict = BulkImportRequestFactory.list_imports_paginated_args(
            limit=limit, pagination_token=pagination_token, **kwargs
        )
        return await self.__import_operations_api.list_imports(**args_dict)

    async def describe_import(self, id: str) -> ImportModel:
        """
        describe_import is used to get detailed information about a specific import operation.

        Args:
            id (str): The id of the import operation. This value is returned when
            starting an import, and can be looked up using list_imports.

        Returns:
            ImportModel: An object containing operation id, status, and other details.
        """
        args = BulkImportRequestFactory.describe_import_args(id=id)
        return await self.__import_operations_api.describe_bulk_import(**args)

    async def cancel_import(self, id: str):
        """Cancel an import operation.

        Args:
            id (str): The id of the import operation to cancel.
        """
        args = BulkImportRequestFactory.cancel_import_args(id=id)
        return await self.__import_operations_api.cancel_import(**args)
