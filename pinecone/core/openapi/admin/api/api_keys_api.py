"""
Pinecone Admin API

Provides an API for managing a Pinecone organization and its resources.   # noqa: E501

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
from pinecone.core.openapi.admin.model.api_key import APIKey
from pinecone.core.openapi.admin.model.api_key_with_secret import APIKeyWithSecret
from pinecone.core.openapi.admin.model.create_api_key_request import CreateAPIKeyRequest
from pinecone.core.openapi.admin.model.error_response import ErrorResponse
from pinecone.core.openapi.admin.model.list_api_keys_response import ListApiKeysResponse
from pinecone.core.openapi.admin.model.update_api_key_request import UpdateAPIKeyRequest


class APIKeysApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __create_api_key(
            self,
            project_id,
            create_api_key_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> APIKeyWithSecret | ApplyResult[APIKeyWithSecret]:
            """Create an API key  # noqa: E501

            Create a new API key for a project. Developers can use the API key to authenticate requests to Pinecone's Data Plane and Control Plane APIs.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_api_key(project_id, create_api_key_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                project_id (str): Project ID
                create_api_key_request (CreateAPIKeyRequest): The details of the new API key.
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
                APIKeyWithSecret
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["project_id"] = project_id
            kwargs["create_api_key_request"] = create_api_key_request
            return cast(
                APIKeyWithSecret | ApplyResult[APIKeyWithSecret], self.call_with_http_info(**kwargs)
            )

        self.create_api_key = _Endpoint(
            settings={
                "response_type": (APIKeyWithSecret,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}/api-keys",
                "operation_id": "create_api_key",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "project_id", "create_api_key_request"],
                "required": ["x_pinecone_api_version", "project_id", "create_api_key_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "project_id": (str,),
                    "create_api_key_request": (CreateAPIKeyRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "project_id": "project_id",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "project_id": "path",
                    "create_api_key_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_api_key,
        )

        def __delete_api_key(
            self,
            api_key_id,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> None:
            """Delete an API key  # noqa: E501

            Delete an API key from a project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_api_key(api_key_id, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                api_key_id (str): API key ID
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
                None
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["api_key_id"] = api_key_id
            return cast(None, self.call_with_http_info(**kwargs))

        self.delete_api_key = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/api-keys/{api_key_id}",
                "operation_id": "delete_api_key",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "api_key_id"],
                "required": ["x_pinecone_api_version", "api_key_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "api_key_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "api_key_id": "api_key_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "api_key_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_api_key,
        )

        def __fetch_api_key(
            self,
            api_key_id,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> APIKey | ApplyResult[APIKey]:
            """Get API key details  # noqa: E501

            Get the details of an API key, excluding the API key secret.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.fetch_api_key(api_key_id, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                api_key_id (str): API key ID
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
                APIKey
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["api_key_id"] = api_key_id
            return cast(APIKey | ApplyResult[APIKey], self.call_with_http_info(**kwargs))

        self.fetch_api_key = _Endpoint(
            settings={
                "response_type": (APIKey,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/api-keys/{api_key_id}",
                "operation_id": "fetch_api_key",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "api_key_id"],
                "required": ["x_pinecone_api_version", "api_key_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "api_key_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "api_key_id": "api_key_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "api_key_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_api_key,
        )

        def __list_project_api_keys(
            self,
            project_id,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> ListApiKeysResponse | ApplyResult[ListApiKeysResponse]:
            """List API keys  # noqa: E501

            List all API keys in a project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_project_api_keys(project_id, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                project_id (str): Project ID
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
                ListApiKeysResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["project_id"] = project_id
            return cast(
                ListApiKeysResponse | ApplyResult[ListApiKeysResponse],
                self.call_with_http_info(**kwargs),
            )

        self.list_project_api_keys = _Endpoint(
            settings={
                "response_type": (ListApiKeysResponse,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}/api-keys",
                "operation_id": "list_project_api_keys",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "project_id"],
                "required": ["x_pinecone_api_version", "project_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "project_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "project_id": "project_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "project_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_project_api_keys,
        )

        def __update_api_key(
            self,
            api_key_id,
            update_api_key_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> APIKey | ApplyResult[APIKey]:
            """Update an API key  # noqa: E501

            Update the name and roles of an API key.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.update_api_key(api_key_id, update_api_key_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                api_key_id (str): API key ID
                update_api_key_request (UpdateAPIKeyRequest): Updated name and roles for the API key.
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
                APIKey
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["api_key_id"] = api_key_id
            kwargs["update_api_key_request"] = update_api_key_request
            return cast(APIKey | ApplyResult[APIKey], self.call_with_http_info(**kwargs))

        self.update_api_key = _Endpoint(
            settings={
                "response_type": (APIKey,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/api-keys/{api_key_id}",
                "operation_id": "update_api_key",
                "http_method": "PATCH",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "api_key_id", "update_api_key_request"],
                "required": ["x_pinecone_api_version", "api_key_id", "update_api_key_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "api_key_id": (str,),
                    "update_api_key_request": (UpdateAPIKeyRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "api_key_id": "api_key_id",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "api_key_id": "path",
                    "update_api_key_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__update_api_key,
        )


class AsyncioAPIKeysApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __create_api_key(
            self, project_id, create_api_key_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> APIKeyWithSecret:
            """Create an API key  # noqa: E501

            Create a new API key for a project. Developers can use the API key to authenticate requests to Pinecone's Data Plane and Control Plane APIs.   # noqa: E501


            Args:
                project_id (str): Project ID
                create_api_key_request (CreateAPIKeyRequest): The details of the new API key.
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
                APIKeyWithSecret
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["project_id"] = project_id
            kwargs["create_api_key_request"] = create_api_key_request
            return cast(APIKeyWithSecret, await self.call_with_http_info(**kwargs))

        self.create_api_key = _AsyncioEndpoint(
            settings={
                "response_type": (APIKeyWithSecret,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}/api-keys",
                "operation_id": "create_api_key",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "project_id", "create_api_key_request"],
                "required": ["x_pinecone_api_version", "project_id", "create_api_key_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "project_id": (str,),
                    "create_api_key_request": (CreateAPIKeyRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "project_id": "project_id",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "project_id": "path",
                    "create_api_key_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_api_key,
        )

        async def __delete_api_key(
            self, api_key_id, x_pinecone_api_version="2025-10", **kwargs
        ) -> None:
            """Delete an API key  # noqa: E501

            Delete an API key from a project.  # noqa: E501


            Args:
                api_key_id (str): API key ID
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
                None
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["api_key_id"] = api_key_id
            return cast(None, await self.call_with_http_info(**kwargs))

        self.delete_api_key = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/api-keys/{api_key_id}",
                "operation_id": "delete_api_key",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "api_key_id"],
                "required": ["x_pinecone_api_version", "api_key_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "api_key_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "api_key_id": "api_key_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "api_key_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_api_key,
        )

        async def __fetch_api_key(
            self, api_key_id, x_pinecone_api_version="2025-10", **kwargs
        ) -> APIKey:
            """Get API key details  # noqa: E501

            Get the details of an API key, excluding the API key secret.  # noqa: E501


            Args:
                api_key_id (str): API key ID
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
                APIKey
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["api_key_id"] = api_key_id
            return cast(APIKey, await self.call_with_http_info(**kwargs))

        self.fetch_api_key = _AsyncioEndpoint(
            settings={
                "response_type": (APIKey,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/api-keys/{api_key_id}",
                "operation_id": "fetch_api_key",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "api_key_id"],
                "required": ["x_pinecone_api_version", "api_key_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "api_key_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "api_key_id": "api_key_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "api_key_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_api_key,
        )

        async def __list_project_api_keys(
            self, project_id, x_pinecone_api_version="2025-10", **kwargs
        ) -> ListApiKeysResponse:
            """List API keys  # noqa: E501

            List all API keys in a project.  # noqa: E501


            Args:
                project_id (str): Project ID
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
                ListApiKeysResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["project_id"] = project_id
            return cast(ListApiKeysResponse, await self.call_with_http_info(**kwargs))

        self.list_project_api_keys = _AsyncioEndpoint(
            settings={
                "response_type": (ListApiKeysResponse,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}/api-keys",
                "operation_id": "list_project_api_keys",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "project_id"],
                "required": ["x_pinecone_api_version", "project_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "project_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "project_id": "project_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "project_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_project_api_keys,
        )

        async def __update_api_key(
            self, api_key_id, update_api_key_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> APIKey:
            """Update an API key  # noqa: E501

            Update the name and roles of an API key.   # noqa: E501


            Args:
                api_key_id (str): API key ID
                update_api_key_request (UpdateAPIKeyRequest): Updated name and roles for the API key.
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
                APIKey
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["api_key_id"] = api_key_id
            kwargs["update_api_key_request"] = update_api_key_request
            return cast(APIKey, await self.call_with_http_info(**kwargs))

        self.update_api_key = _AsyncioEndpoint(
            settings={
                "response_type": (APIKey,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/api-keys/{api_key_id}",
                "operation_id": "update_api_key",
                "http_method": "PATCH",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "api_key_id", "update_api_key_request"],
                "required": ["x_pinecone_api_version", "api_key_id", "update_api_key_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "api_key_id": (str,),
                    "update_api_key_request": (UpdateAPIKeyRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "api_key_id": "api_key_id",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "api_key_id": "path",
                    "update_api_key_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__update_api_key,
        )
