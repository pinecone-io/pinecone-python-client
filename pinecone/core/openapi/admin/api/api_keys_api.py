"""
Pinecone Admin API

Provides an API for managing a Pinecone organization and its resources.   # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: 2025-04
Contact: support@pinecone.io
"""

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
from pinecone.core.openapi.admin.model.inline_response2001 import InlineResponse2001
from pinecone.core.openapi.admin.model.inline_response401 import InlineResponse401


class APIKeysApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __create_api_key(
            self, project_id, create_api_key_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Create an API key  # noqa: E501

            Create a new API key for a project. Developers can use the API key to authenticate requests to Pinecone's Data Plane and Control Plane APIs.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_api_key(project_id, create_api_key_request, async_req=True)
            >>> result = thread.get()

            Args:
                project_id (str): Project ID
                create_api_key_request (CreateAPIKeyRequest): The details of the new API key.

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
            kwargs["project_id"] = project_id
            kwargs["create_api_key_request"] = create_api_key_request
            return self.call_with_http_info(**kwargs)

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
                "all": ["project_id", "create_api_key_request"],
                "required": ["project_id", "create_api_key_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "project_id": (str,),
                    "create_api_key_request": (CreateAPIKeyRequest,),
                },
                "attribute_map": {"project_id": "project_id"},
                "location_map": {"project_id": "path", "create_api_key_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_api_key,
        )

        def __delete_api_key(self, api_key_id, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Delete an API key  # noqa: E501

            Delete an API key from a project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_api_key(api_key_id, async_req=True)
            >>> result = thread.get()

            Args:
                api_key_id (str): API key ID

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
            kwargs["api_key_id"] = api_key_id
            return self.call_with_http_info(**kwargs)

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
                "all": ["api_key_id"],
                "required": ["api_key_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"api_key_id": (str,)},
                "attribute_map": {"api_key_id": "api_key_id"},
                "location_map": {"api_key_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_api_key,
        )

        def __fetch_api_key(self, api_key_id, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Get API key details  # noqa: E501

            Get the details of an API key, excluding the API key secret.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.fetch_api_key(api_key_id, async_req=True)
            >>> result = thread.get()

            Args:
                api_key_id (str): API key ID

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
            kwargs["api_key_id"] = api_key_id
            return self.call_with_http_info(**kwargs)

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
                "all": ["api_key_id"],
                "required": ["api_key_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"api_key_id": (str,)},
                "attribute_map": {"api_key_id": "api_key_id"},
                "location_map": {"api_key_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_api_key,
        )

        def __list_api_keys(self, project_id, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List API keys  # noqa: E501

            List all API keys in a project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_api_keys(project_id, async_req=True)
            >>> result = thread.get()

            Args:
                project_id (str): Project ID

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
                InlineResponse2001
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["project_id"] = project_id
            return self.call_with_http_info(**kwargs)

        self.list_api_keys = _Endpoint(
            settings={
                "response_type": (InlineResponse2001,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}/api-keys",
                "operation_id": "list_api_keys",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["project_id"],
                "required": ["project_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"project_id": (str,)},
                "attribute_map": {"project_id": "project_id"},
                "location_map": {"project_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_api_keys,
        )


class AsyncioAPIKeysApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __create_api_key(self, project_id, create_api_key_request, **kwargs):
            """Create an API key  # noqa: E501

            Create a new API key for a project. Developers can use the API key to authenticate requests to Pinecone's Data Plane and Control Plane APIs.   # noqa: E501


            Args:
                project_id (str): Project ID
                create_api_key_request (CreateAPIKeyRequest): The details of the new API key.

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
            kwargs["project_id"] = project_id
            kwargs["create_api_key_request"] = create_api_key_request
            return await self.call_with_http_info(**kwargs)

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
                "all": ["project_id", "create_api_key_request"],
                "required": ["project_id", "create_api_key_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "project_id": (str,),
                    "create_api_key_request": (CreateAPIKeyRequest,),
                },
                "attribute_map": {"project_id": "project_id"},
                "location_map": {"project_id": "path", "create_api_key_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_api_key,
        )

        async def __delete_api_key(self, api_key_id, **kwargs):
            """Delete an API key  # noqa: E501

            Delete an API key from a project.  # noqa: E501


            Args:
                api_key_id (str): API key ID

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
            kwargs["api_key_id"] = api_key_id
            return await self.call_with_http_info(**kwargs)

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
                "all": ["api_key_id"],
                "required": ["api_key_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"api_key_id": (str,)},
                "attribute_map": {"api_key_id": "api_key_id"},
                "location_map": {"api_key_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_api_key,
        )

        async def __fetch_api_key(self, api_key_id, **kwargs):
            """Get API key details  # noqa: E501

            Get the details of an API key, excluding the API key secret.  # noqa: E501


            Args:
                api_key_id (str): API key ID

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
            kwargs["api_key_id"] = api_key_id
            return await self.call_with_http_info(**kwargs)

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
                "all": ["api_key_id"],
                "required": ["api_key_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"api_key_id": (str,)},
                "attribute_map": {"api_key_id": "api_key_id"},
                "location_map": {"api_key_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_api_key,
        )

        async def __list_api_keys(self, project_id, **kwargs):
            """List API keys  # noqa: E501

            List all API keys in a project.  # noqa: E501


            Args:
                project_id (str): Project ID

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
                InlineResponse2001
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["project_id"] = project_id
            return await self.call_with_http_info(**kwargs)

        self.list_api_keys = _AsyncioEndpoint(
            settings={
                "response_type": (InlineResponse2001,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}/api-keys",
                "operation_id": "list_api_keys",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["project_id"],
                "required": ["project_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"project_id": (str,)},
                "attribute_map": {"project_id": "project_id"},
                "location_map": {"project_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_api_keys,
        )
