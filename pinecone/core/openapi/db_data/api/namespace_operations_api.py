"""
Pinecone Data Plane API

Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: 2025-10
Contact: support@pinecone.io
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, cast
from multiprocessing.pool import ApplyResult

from pinecone.openapi_support import ApiClient, AsyncioApiClient
from pinecone.openapi_support.endpoint_utils import (
    ExtraOpenApiKwargsTypedDict,
    KwargsWithOpenApiKwargDefaultsTypedDict,
)
from pinecone.openapi_support.endpoint import Endpoint as _Endpoint, ExtraOpenApiKwargsTypedDict
from pinecone.openapi_support.asyncio_endpoint import AsyncioEndpoint as _AsyncioEndpoint
from pinecone.openapi_support.model_utils import (  # noqa: F401
    date,
    datetime,
    file_type,
    none_type,
    validate_and_convert_types,
)
from pinecone.core.openapi.db_data.model.create_namespace_request import CreateNamespaceRequest
from pinecone.core.openapi.db_data.model.list_namespaces_response import ListNamespacesResponse
from pinecone.core.openapi.db_data.model.namespace_description import NamespaceDescription
from pinecone.core.openapi.db_data.model.rpc_status import RpcStatus


class NamespaceOperationsApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __create_namespace(
            self,
            create_namespace_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> NamespaceDescription | ApplyResult[NamespaceDescription]:
            """Create a namespace  # noqa: E501

            Create a namespace in a serverless index.  For guidance and examples, see [Manage namespaces](https://docs.pinecone.io/guides/manage-data/manage-namespaces).  **Note:** This operation is not supported for pod-based indexes.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_namespace(create_namespace_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                create_namespace_request (CreateNamespaceRequest):
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

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
                async_req (bool): execute request asynchronously

            Returns:
                NamespaceDescription
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["create_namespace_request"] = create_namespace_request
            return cast(
                NamespaceDescription | ApplyResult[NamespaceDescription],
                self.call_with_http_info(**kwargs),
            )

        self.create_namespace = _Endpoint(
            settings={
                "response_type": (NamespaceDescription,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces",
                "operation_id": "create_namespace",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "create_namespace_request"],
                "required": ["x_pinecone_api_version", "create_namespace_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "create_namespace_request": (CreateNamespaceRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "create_namespace_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_namespace,
        )

        def __delete_namespace(
            self, namespace, x_pinecone_api_version="2025-10", **kwargs: ExtraOpenApiKwargsTypedDict
        ) -> Dict[str, Any] | ApplyResult[Dict[str, Any]]:
            """Delete a namespace  # noqa: E501

            Delete a namespace from a serverless index. Deleting a namespace is irreversible; all data in the namespace is permanently deleted.  For guidance and examples, see [Manage namespaces](https://docs.pinecone.io/guides/manage-data/manage-namespaces).  **Note:** This operation is not supported for pod-based indexes.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_namespace(namespace, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): The namespace to delete.
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

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
                async_req (bool): execute request asynchronously

            Returns:
                Dict[str, Any]
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["namespace"] = namespace
            return cast(
                Dict[str, Any] | ApplyResult[Dict[str, Any]], self.call_with_http_info(**kwargs)
            )

        self.delete_namespace = _Endpoint(
            settings={
                "response_type": (Dict[str, Any],),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}",
                "operation_id": "delete_namespace",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace"],
                "required": ["x_pinecone_api_version", "namespace"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "namespace": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {"x_pinecone_api_version": "header", "namespace": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_namespace,
        )

        def __describe_namespace(
            self, namespace, x_pinecone_api_version="2025-10", **kwargs: ExtraOpenApiKwargsTypedDict
        ) -> NamespaceDescription | ApplyResult[NamespaceDescription]:
            """Describe a namespace  # noqa: E501

            Describe a namespace in a serverless index, including the total number of vectors in the namespace.  For guidance and examples, see [Manage namespaces](https://docs.pinecone.io/guides/manage-data/manage-namespaces).  **Note:** This operation is not supported for pod-based indexes.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_namespace(namespace, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): The namespace to describe.
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

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
                async_req (bool): execute request asynchronously

            Returns:
                NamespaceDescription
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["namespace"] = namespace
            return cast(
                NamespaceDescription | ApplyResult[NamespaceDescription],
                self.call_with_http_info(**kwargs),
            )

        self.describe_namespace = _Endpoint(
            settings={
                "response_type": (NamespaceDescription,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}",
                "operation_id": "describe_namespace",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace"],
                "required": ["x_pinecone_api_version", "namespace"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "namespace": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {"x_pinecone_api_version": "header", "namespace": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_namespace,
        )

        def __list_namespaces_operation(
            self, x_pinecone_api_version="2025-10", **kwargs: ExtraOpenApiKwargsTypedDict
        ) -> ListNamespacesResponse | ApplyResult[ListNamespacesResponse]:
            """List namespaces  # noqa: E501

            List all namespaces in a serverless index.  Up to 100 namespaces are returned at a time by default, in sorted order (bitwise “C” collation). If the `limit` parameter is set, up to that number of namespaces are returned instead. Whenever there are additional namespaces to return, the response also includes a `pagination_token` that you can use to get the next batch of namespaces. When the response does not include a `pagination_token`, there are no more namespaces to return.  For guidance and examples, see [Manage namespaces](https://docs.pinecone.io/guides/manage-data/manage-namespaces).  **Note:** This operation is not supported for pod-based indexes.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_namespaces_operation(x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

            Keyword Args:
                limit (int): Max number namespaces to return per page. [optional]
                pagination_token (str): Pagination token to continue a previous listing operation. [optional]
                prefix (str): Prefix of the namespaces to list. Acts as a filter to return only namespaces that start with this prefix. [optional]
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
                async_req (bool): execute request asynchronously

            Returns:
                ListNamespacesResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            return cast(
                ListNamespacesResponse | ApplyResult[ListNamespacesResponse],
                self.call_with_http_info(**kwargs),
            )

        self.list_namespaces_operation = _Endpoint(
            settings={
                "response_type": (ListNamespacesResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces",
                "operation_id": "list_namespaces_operation",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "limit", "pagination_token", "prefix"],
                "required": ["x_pinecone_api_version"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "limit": (int,),
                    "pagination_token": (str,),
                    "prefix": (str,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "limit": "limit",
                    "pagination_token": "paginationToken",
                    "prefix": "prefix",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "limit": "query",
                    "pagination_token": "query",
                    "prefix": "query",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_namespaces_operation,
        )


class AsyncioNamespaceOperationsApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __create_namespace(
            self, create_namespace_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> NamespaceDescription:
            """Create a namespace  # noqa: E501

            Create a namespace in a serverless index.  For guidance and examples, see [Manage namespaces](https://docs.pinecone.io/guides/manage-data/manage-namespaces).  **Note:** This operation is not supported for pod-based indexes.  # noqa: E501


            Args:
                create_namespace_request (CreateNamespaceRequest):
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

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

            Returns:
                NamespaceDescription
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["create_namespace_request"] = create_namespace_request
            return cast(NamespaceDescription, await self.call_with_http_info(**kwargs))

        self.create_namespace = _AsyncioEndpoint(
            settings={
                "response_type": (NamespaceDescription,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces",
                "operation_id": "create_namespace",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "create_namespace_request"],
                "required": ["x_pinecone_api_version", "create_namespace_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "create_namespace_request": (CreateNamespaceRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "create_namespace_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_namespace,
        )

        async def __delete_namespace(
            self, namespace, x_pinecone_api_version="2025-10", **kwargs
        ) -> Dict[str, Any]:
            """Delete a namespace  # noqa: E501

            Delete a namespace from a serverless index. Deleting a namespace is irreversible; all data in the namespace is permanently deleted.  For guidance and examples, see [Manage namespaces](https://docs.pinecone.io/guides/manage-data/manage-namespaces).  **Note:** This operation is not supported for pod-based indexes.  # noqa: E501


            Args:
                namespace (str): The namespace to delete.
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

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

            Returns:
                Dict[str, Any]
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["namespace"] = namespace
            return cast(Dict[str, Any], await self.call_with_http_info(**kwargs))

        self.delete_namespace = _AsyncioEndpoint(
            settings={
                "response_type": (Dict[str, Any],),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}",
                "operation_id": "delete_namespace",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace"],
                "required": ["x_pinecone_api_version", "namespace"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "namespace": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {"x_pinecone_api_version": "header", "namespace": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_namespace,
        )

        async def __describe_namespace(
            self, namespace, x_pinecone_api_version="2025-10", **kwargs
        ) -> NamespaceDescription:
            """Describe a namespace  # noqa: E501

            Describe a namespace in a serverless index, including the total number of vectors in the namespace.  For guidance and examples, see [Manage namespaces](https://docs.pinecone.io/guides/manage-data/manage-namespaces).  **Note:** This operation is not supported for pod-based indexes.  # noqa: E501


            Args:
                namespace (str): The namespace to describe.
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

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

            Returns:
                NamespaceDescription
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["namespace"] = namespace
            return cast(NamespaceDescription, await self.call_with_http_info(**kwargs))

        self.describe_namespace = _AsyncioEndpoint(
            settings={
                "response_type": (NamespaceDescription,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}",
                "operation_id": "describe_namespace",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace"],
                "required": ["x_pinecone_api_version", "namespace"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "namespace": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {"x_pinecone_api_version": "header", "namespace": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_namespace,
        )

        async def __list_namespaces_operation(
            self, x_pinecone_api_version="2025-10", **kwargs
        ) -> ListNamespacesResponse:
            """List namespaces  # noqa: E501

            List all namespaces in a serverless index.  Up to 100 namespaces are returned at a time by default, in sorted order (bitwise “C” collation). If the `limit` parameter is set, up to that number of namespaces are returned instead. Whenever there are additional namespaces to return, the response also includes a `pagination_token` that you can use to get the next batch of namespaces. When the response does not include a `pagination_token`, there are no more namespaces to return.  For guidance and examples, see [Manage namespaces](https://docs.pinecone.io/guides/manage-data/manage-namespaces).  **Note:** This operation is not supported for pod-based indexes.  # noqa: E501


            Args:
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

            Keyword Args:
                limit (int): Max number namespaces to return per page. [optional]
                pagination_token (str): Pagination token to continue a previous listing operation. [optional]
                prefix (str): Prefix of the namespaces to list. Acts as a filter to return only namespaces that start with this prefix. [optional]
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

            Returns:
                ListNamespacesResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            return cast(ListNamespacesResponse, await self.call_with_http_info(**kwargs))

        self.list_namespaces_operation = _AsyncioEndpoint(
            settings={
                "response_type": (ListNamespacesResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces",
                "operation_id": "list_namespaces_operation",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "limit", "pagination_token", "prefix"],
                "required": ["x_pinecone_api_version"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "limit": (int,),
                    "pagination_token": (str,),
                    "prefix": (str,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "limit": "limit",
                    "pagination_token": "paginationToken",
                    "prefix": "prefix",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "limit": "query",
                    "pagination_token": "query",
                    "prefix": "query",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_namespaces_operation,
        )
