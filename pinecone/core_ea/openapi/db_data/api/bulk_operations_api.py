"""
    Pinecone Data Plane API

    Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

    The version of the OpenAPI document: 2024-10
    Contact: support@pinecone.io
    Generated by: https://openapi-generator.tech
"""

import re  # noqa: F401
import sys  # noqa: F401

from pinecone.core_ea.openapi.shared.api_client import ApiClient, Endpoint as _Endpoint
from pinecone.core_ea.openapi.shared.model_utils import (  # noqa: F401
    check_allowed_values,
    check_validations,
    date,
    datetime,
    file_type,
    none_type,
    validate_and_convert_types,
)
from pinecone.core_ea.openapi.db_data.model.import_list_response import ImportListResponse
from pinecone.core_ea.openapi.db_data.model.import_model import ImportModel
from pinecone.core_ea.openapi.db_data.model.rpc_status import RpcStatus
from pinecone.core_ea.openapi.db_data.model.start_import_request import StartImportRequest
from pinecone.core_ea.openapi.db_data.model.start_import_response import StartImportResponse


class BulkOperationsApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __cancel_import(self, operation_id, **kwargs):
            """Cancel an ongoing import operation  # noqa: E501

            The `cancel_import` operation cancels an import operation if it is not yet finished. It has no effect if the operation is already finished. For guidance and examples, see [Import data](https://docs.pinecone.io/guides/data/import-data).   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.cancel_import(operation_id, async_req=True)
            >>> result = thread.get()

            Args:
                operation_id (str):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                {str: (bool, date, datetime, dict, float, int, list, str, none_type)}
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
            kwargs["operation_id"] = operation_id
            return self.call_with_http_info(**kwargs)

        self.cancel_import = _Endpoint(
            settings={
                "response_type": ({str: (bool, date, datetime, dict, float, int, list, str, none_type)},),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports/{operation_id}",
                "operation_id": "cancel_import",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": [
                    "operation_id",
                ],
                "required": [
                    "operation_id",
                ],
                "nullable": [],
                "enum": [],
                "validation": [
                    "operation_id",
                ],
            },
            root_map={
                "validations": {
                    ("operation_id",): {
                        "max_length": 1000,
                        "min_length": 1,
                    },
                },
                "allowed_values": {},
                "openapi_types": {
                    "operation_id": (str,),
                },
                "attribute_map": {
                    "operation_id": "operation_id",
                },
                "location_map": {
                    "operation_id": "path",
                },
                "collection_format_map": {},
            },
            headers_map={
                "accept": ["application/json"],
                "content_type": [],
            },
            api_client=api_client,
            callable=__cancel_import,
        )

        def __describe_import(self, operation_id, **kwargs):
            """Describe a specific import operation  # noqa: E501

            The `describe_import` operation returns details of a specific import operation. For guidance and examples, see [Import data](https://docs.pinecone.io/guides/data/import-data).   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_import(operation_id, async_req=True)
            >>> result = thread.get()

            Args:
                operation_id (str):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                ImportModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
            kwargs["operation_id"] = operation_id
            return self.call_with_http_info(**kwargs)

        self.describe_import = _Endpoint(
            settings={
                "response_type": (ImportModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports/{operation_id}",
                "operation_id": "describe_import",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": [
                    "operation_id",
                ],
                "required": [
                    "operation_id",
                ],
                "nullable": [],
                "enum": [],
                "validation": [
                    "operation_id",
                ],
            },
            root_map={
                "validations": {
                    ("operation_id",): {
                        "max_length": 1000,
                        "min_length": 1,
                    },
                },
                "allowed_values": {},
                "openapi_types": {
                    "operation_id": (str,),
                },
                "attribute_map": {
                    "operation_id": "operation_id",
                },
                "location_map": {
                    "operation_id": "path",
                },
                "collection_format_map": {},
            },
            headers_map={
                "accept": ["application/json"],
                "content_type": [],
            },
            api_client=api_client,
            callable=__describe_import,
        )

        def __list_imports(self, **kwargs):
            """List all recent and ongoing import operations  # noqa: E501

            The `list_imports` operation lists all recent and ongoing import operations. For guidance and examples, see [Import data](https://docs.pinecone.io/guides/data/import-data).   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_imports(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                limit (int): Max number of operations to return per page.. [optional]
                pagination_token (str): Pagination token to continue a previous listing operation.. [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                ImportListResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
            return self.call_with_http_info(**kwargs)

        self.list_imports = _Endpoint(
            settings={
                "response_type": (ImportListResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports",
                "operation_id": "list_imports",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": [
                    "limit",
                    "pagination_token",
                ],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [
                    "limit",
                ],
            },
            root_map={
                "validations": {
                    ("limit",): {
                        "inclusive_maximum": 100,
                        "inclusive_minimum": 1,
                    },
                },
                "allowed_values": {},
                "openapi_types": {
                    "limit": (int,),
                    "pagination_token": (str,),
                },
                "attribute_map": {
                    "limit": "limit",
                    "pagination_token": "paginationToken",
                },
                "location_map": {
                    "limit": "query",
                    "pagination_token": "query",
                },
                "collection_format_map": {},
            },
            headers_map={
                "accept": ["application/json"],
                "content_type": [],
            },
            api_client=api_client,
            callable=__list_imports,
        )

        def __start_import(self, start_import_request, **kwargs):
            """Start import  # noqa: E501

            The `start_import` operation starts an asynchronous import of vectors from blob storage into an index. For guidance and examples, see [Import data](https://docs.pinecone.io/guides/data/import-data).   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.start_import(start_import_request, async_req=True)
            >>> result = thread.get()

            Args:
                start_import_request (StartImportRequest):

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                StartImportResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
            kwargs["start_import_request"] = start_import_request
            return self.call_with_http_info(**kwargs)

        self.start_import = _Endpoint(
            settings={
                "response_type": (StartImportResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/bulk/imports",
                "operation_id": "start_import",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": [
                    "start_import_request",
                ],
                "required": [
                    "start_import_request",
                ],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "start_import_request": (StartImportRequest,),
                },
                "attribute_map": {},
                "location_map": {
                    "start_import_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__start_import,
        )
