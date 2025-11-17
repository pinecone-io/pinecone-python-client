"""
Pinecone OAuth API

Provides an API for authenticating with Pinecone.   # noqa: E501

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
from pinecone.core.openapi.oauth.model.error_response import ErrorResponse
from pinecone.core.openapi.oauth.model.token_request import TokenRequest
from pinecone.core.openapi.oauth.model.token_response import TokenResponse


class OAuthApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __get_token(
            self,
            token_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> TokenResponse | ApplyResult[TokenResponse]:
            """Create an access token  # noqa: E501

            Obtain an access token for a service account using the OAuth2 client credentials flow. An access token is needed to authorize requests to the Pinecone Admin API. The host domain for OAuth endpoints is `login.pinecone.io`.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.get_token(token_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                token_request (TokenRequest): A request to exchange client credentials for an access token.
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
                TokenResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["token_request"] = token_request
            return cast(
                TokenResponse | ApplyResult[TokenResponse], self.call_with_http_info(**kwargs)
            )

        self.get_token = _Endpoint(
            settings={
                "response_type": (TokenResponse,),
                "auth": [],
                "endpoint_path": "/oauth/token",
                "operation_id": "get_token",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "token_request"],
                "required": ["x_pinecone_api_version", "token_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "token_request": (TokenRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header", "token_request": "body"},
                "collection_format_map": {},
            },
            headers_map={
                "accept": ["application/json"],
                "content_type": ["application/json", "application/x-www-form-urlencoded"],
            },
            api_client=api_client,
            callable=__get_token,
        )


class AsyncioOAuthApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __get_token(
            self, token_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> TokenResponse:
            """Create an access token  # noqa: E501

            Obtain an access token for a service account using the OAuth2 client credentials flow. An access token is needed to authorize requests to the Pinecone Admin API. The host domain for OAuth endpoints is `login.pinecone.io`.   # noqa: E501


            Args:
                token_request (TokenRequest): A request to exchange client credentials for an access token.
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
                TokenResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["token_request"] = token_request
            return cast(TokenResponse, await self.call_with_http_info(**kwargs))

        self.get_token = _AsyncioEndpoint(
            settings={
                "response_type": (TokenResponse,),
                "auth": [],
                "endpoint_path": "/oauth/token",
                "operation_id": "get_token",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "token_request"],
                "required": ["x_pinecone_api_version", "token_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "token_request": (TokenRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header", "token_request": "body"},
                "collection_format_map": {},
            },
            headers_map={
                "accept": ["application/json"],
                "content_type": ["application/json", "application/x-www-form-urlencoded"],
            },
            api_client=api_client,
            callable=__get_token,
        )
