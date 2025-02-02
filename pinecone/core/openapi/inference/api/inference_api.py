"""
Pinecone Inference API

Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: 2025-01
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
from pinecone.core.openapi.inference.model.embed_request import EmbedRequest
from pinecone.core.openapi.inference.model.embeddings_list import EmbeddingsList
from pinecone.core.openapi.inference.model.error_response import ErrorResponse
from pinecone.core.openapi.inference.model.rerank_request import RerankRequest
from pinecone.core.openapi.inference.model.rerank_result import RerankResult


class InferenceApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __embed(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Embed data  # noqa: E501

            Generate embeddings for input data.  For guidance and examples, see [Generate embeddings](https://docs.pinecone.io/guides/inference/generate-embeddings).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.embed(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                embed_request (EmbedRequest): Generate embeddings for inputs. [optional]
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
                EmbeddingsList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.embed = _Endpoint(
            settings={
                "response_type": (EmbeddingsList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/embed",
                "operation_id": "embed",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["embed_request"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"embed_request": (EmbedRequest,)},
                "attribute_map": {},
                "location_map": {"embed_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__embed,
        )

        def __rerank(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Rerank documents  # noqa: E501

            Rerank documents according to their relevance to a query.  For guidance and examples, see [Rerank documents](https://docs.pinecone.io/guides/inference/rerank).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.rerank(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                rerank_request (RerankRequest): Rerank documents for the given query [optional]
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
                RerankResult
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.rerank = _Endpoint(
            settings={
                "response_type": (RerankResult,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/rerank",
                "operation_id": "rerank",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["rerank_request"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"rerank_request": (RerankRequest,)},
                "attribute_map": {},
                "location_map": {"rerank_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__rerank,
        )


class AsyncioInferenceApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __embed(self, **kwargs):
            """Embed data  # noqa: E501

            Generate embeddings for input data.  For guidance and examples, see [Generate embeddings](https://docs.pinecone.io/guides/inference/generate-embeddings).  # noqa: E501



            Keyword Args:
                embed_request (EmbedRequest): Generate embeddings for inputs. [optional]
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
                EmbeddingsList
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.embed = _AsyncioEndpoint(
            settings={
                "response_type": (EmbeddingsList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/embed",
                "operation_id": "embed",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["embed_request"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"embed_request": (EmbedRequest,)},
                "attribute_map": {},
                "location_map": {"embed_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__embed,
        )

        async def __rerank(self, **kwargs):
            """Rerank documents  # noqa: E501

            Rerank documents according to their relevance to a query.  For guidance and examples, see [Rerank documents](https://docs.pinecone.io/guides/inference/rerank).  # noqa: E501



            Keyword Args:
                rerank_request (RerankRequest): Rerank documents for the given query [optional]
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
                RerankResult
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.rerank = _AsyncioEndpoint(
            settings={
                "response_type": (RerankResult,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/rerank",
                "operation_id": "rerank",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["rerank_request"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"rerank_request": (RerankRequest,)},
                "attribute_map": {},
                "location_map": {"rerank_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__rerank,
        )
