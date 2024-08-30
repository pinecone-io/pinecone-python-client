import warnings

from typing import Optional, Union, Literal, Iterator, List

from pinecone.config.config import ConfigBuilder
from pinecone.core_ea.openapi.db_data import ApiClient
from pinecone.core_ea.openapi.db_data.api.bulk_operations_api import BulkOperationsApi
from pinecone.core_ea.openapi.shared import API_VERSION

from pinecone.utils import parse_non_empty_args, install_json_repr_override, setup_openapi_client

from pinecone.core_ea.openapi.db_data.models import (
    StartImportRequest,
    StartImportResponse,
    ImportListResponse,
    ImportModel,
    ImportErrorMode as ImportErrorModeClass,
)

for m in [StartImportResponse, ImportListResponse, ImportModel]:
    install_json_repr_override(m)


def prerelease_feature(func):
    def wrapper(*args, **kwargs):
        warnmsg = (
            f"This is a prerelease feature implemented against the {API_VERSION} version of our API. Use with caution."
        )
        warnings.warn(warnmsg)
        return func(*args, **kwargs)

    return wrapper


class ImportFeatureMixin:
    def __init__(self, **kwargs):
        config = ConfigBuilder.build(
            **kwargs,
        )
        openapi_config = ConfigBuilder.build_openapi_config(config, kwargs.get("openapi_config", None))

        if kwargs.get("__import_operations_api", None):
            self.__import_operations_api = kwargs.get("__import_operations_api")
        else:
            self.__import_operations_api = setup_openapi_client(
                api_client_klass=ApiClient,
                api_klass=BulkOperationsApi,
                config=config,
                openapi_config=openapi_config,
                pool_threads=kwargs.get("pool_threads", 1),
                api_version=API_VERSION,
            )

    ImportErrorMode = Literal["CONTINUE", "ABORT"]

    @prerelease_feature
    def start_import(
        self,
        uri: str,
        integration: Optional[str] = None,
        error_mode: Optional[ImportErrorMode] = "CONTINUE",
    ) -> StartImportResponse:
        """Import data from a URI into an index.

        Examples:
            >>> from pinecone import Pinecone
            >>> index = Pinecone().Index('my-index')
            >>> index.start_import("s3://bucket-name/path/to/data.parquet")

        Args:
            uri (str): The URI of the data to import. The URI must start with the scheme of a supported storage provider.
            integration (Optional[str], optional): Defaults to None.
            error_mode: Defaults to "CONTINUE". If set to "CONTINUE", the import operation will continue even if some
                records fail to import. Pass "ABORT" to stop the import operation if any records fail to import.

        Returns:
            StartImportResponse: Contains the id of the import operation.
        """
        args_dict = parse_non_empty_args(
            [
                ("uri", uri),
                ("integration", integration),
                ("error_mode", ImportErrorModeClass(error_mode=error_mode)),
            ]
        )
        return self.__import_operations_api.start_import(StartImportRequest(**args_dict))

    @prerelease_feature
    def list_imports(self, **kwargs) -> Iterator[List[ImportModel]]:
        """
        The list_imports operation accepts all of the same arguments as list_imports_paginated, and returns a generator that yields
        a list of operations in each page of results. It automatically handles pagination tokens on your
        behalf so you can easily iterate over all results.

        Args:
            limit (Optional[int]): The maximum number of operations to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
        """
        done = False
        while not done:
            results = self.list_imports_paginated(**kwargs)
            if len(results.data) > 0:
                yield results.data

            if results.pagination:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

    @prerelease_feature
    def list_imports_paginated(
        self,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        **kwargs,
    ) -> ImportListResponse:
        """
        The list_imports_paginated operation returns information about import operations.
        It returns operations in a paginated form, with a pagination token to fetch the next page of results.

        Consider using the `list_imports` method to avoid having to handle pagination tokens manually.

        Examples:
            >>> results = index.list_imports_paginated(limit=5)
            >>> [v.id for v in results.operations]
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> next_results = index.list_paginated(limit=5, pagination_token=results.pagination.next)

        Args:
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]

        Returns: ImportListResponse object which contains the list of operations as ImportModel objects, pagination information,
            and usage showing the number of read_units consumed.
        """
        args_dict = parse_non_empty_args(
            [
                ("limit", limit),
                ("pagination_token", pagination_token),
                # ("namespace", namespace),
            ]
        )
        return self.__import_operations_api.list_imports(**args_dict)

    @prerelease_feature
    def describe_import(self, id: str) -> ImportModel:
        """
        describe_import is used to get detailed information about a specific import operation.

        Args:
            id (str): The id of the import operation. This value is returned when
            starting an import, and can be looked up using list_imports.

        Returns:
            ImportModel: An object containing operation id, status, and other details.
        """
        return self.__import_operations_api.describe_import(id=id)

    @prerelease_feature
    def cancel_import(self, id: str):
        """Cancel an import operation.

        Args:
            id (str): The id of the import operation to cancel.
        """
        return self.__import_operations_api.cancel_import(id=id)
