"""
Pinecone Data Plane API

Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: 2026-01.alpha
Contact: support@pinecone.io
"""

from __future__ import annotations

from typing import TYPE_CHECKING
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
from pinecone.core.openapi.db_data.model.document_search_request import DocumentSearchRequest
from pinecone.core.openapi.db_data.model.document_search_response import DocumentSearchResponse
from pinecone.core.openapi.db_data.model.document_upsert_request import DocumentUpsertRequest
from pinecone.core.openapi.db_data.model.document_upsert_response import DocumentUpsertResponse
from pinecone.core.openapi.db_data.model.rpc_status import RpcStatus


class DocumentOperationsApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __search_documents(
            self,
            namespace,
            document_search_request,
            x_pinecone_api_version="2026-01.alpha",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> DocumentSearchResponse | ApplyResult[DocumentSearchResponse]:
            """Search documents  # noqa: E501

            Search documents in a namespace using text search, vector search, or sparse vector search. Results can be filtered by metadata and ranked using the `score_by` parameter. For v0, a single query can only be ranked by: - Pure text query (multiple terms on the same field) - Pure vector query (one field) For guidance and examples, see [Search documents](https://docs.pinecone.io/guides/search/search-overview).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.search_documents(namespace, document_search_request, x_pinecone_api_version="2026-01.alpha", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): The namespace to search.
                document_search_request (DocumentSearchRequest):
                x_pinecone_api_version (str): Required date-based version header Defaults to "2026-01.alpha", must be one of ["2026-01.alpha"]

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
                DocumentSearchResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["namespace"] = namespace
            kwargs["document_search_request"] = document_search_request
            return self.call_with_http_info(**kwargs)

        self.search_documents = _Endpoint(
            settings={
                "response_type": (DocumentSearchResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}/documents/search",
                "operation_id": "search_documents",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace", "document_search_request"],
                "required": ["x_pinecone_api_version", "namespace", "document_search_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "namespace": (str,),
                    "document_search_request": (DocumentSearchRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "namespace": "path",
                    "document_search_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__search_documents,
        )

        def __upsert_documents(
            self,
            namespace,
            document_upsert_request,
            x_pinecone_api_version="2026-01.alpha",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> DocumentUpsertResponse | ApplyResult[DocumentUpsertResponse]:
            """Upsert documents  # noqa: E501

            Upsert flat JSON documents into a namespace. Documents are indexed based on the configured index schema. Vector fields can be user-specified (e.g., `my_vector`) or use the reserved `_values` key. Text fields are indexed based on schema configuration with `full_text_searchable: true`. For guidance and examples, see [Upsert documents](https://docs.pinecone.io/guides/index-data/upsert-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.upsert_documents(namespace, document_upsert_request, x_pinecone_api_version="2026-01.alpha", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): The namespace to upsert documents into.
                document_upsert_request (DocumentUpsertRequest):
                x_pinecone_api_version (str): Required date-based version header Defaults to "2026-01.alpha", must be one of ["2026-01.alpha"]

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
                DocumentUpsertResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["namespace"] = namespace
            kwargs["document_upsert_request"] = document_upsert_request
            return self.call_with_http_info(**kwargs)

        self.upsert_documents = _Endpoint(
            settings={
                "response_type": (DocumentUpsertResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}/documents/upsert",
                "operation_id": "upsert_documents",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace", "document_upsert_request"],
                "required": ["x_pinecone_api_version", "namespace", "document_upsert_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "namespace": (str,),
                    "document_upsert_request": (DocumentUpsertRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "namespace": "path",
                    "document_upsert_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__upsert_documents,
        )


class AsyncioDocumentOperationsApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __search_documents(
            self,
            namespace,
            document_search_request,
            x_pinecone_api_version="2026-01.alpha",
            **kwargs,
        ) -> DocumentSearchResponse:
            """Search documents  # noqa: E501

            Search documents in a namespace using text search, vector search, or sparse vector search. Results can be filtered by metadata and ranked using the `score_by` parameter. For v0, a single query can only be ranked by: - Pure text query (multiple terms on the same field) - Pure vector query (one field) For guidance and examples, see [Search documents](https://docs.pinecone.io/guides/search/search-overview).  # noqa: E501


            Args:
                namespace (str): The namespace to search.
                document_search_request (DocumentSearchRequest):
                x_pinecone_api_version (str): Required date-based version header Defaults to "2026-01.alpha", must be one of ["2026-01.alpha"]

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
                DocumentSearchResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["namespace"] = namespace
            kwargs["document_search_request"] = document_search_request
            return await self.call_with_http_info(**kwargs)

        self.search_documents = _AsyncioEndpoint(
            settings={
                "response_type": (DocumentSearchResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}/documents/search",
                "operation_id": "search_documents",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace", "document_search_request"],
                "required": ["x_pinecone_api_version", "namespace", "document_search_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "namespace": (str,),
                    "document_search_request": (DocumentSearchRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "namespace": "path",
                    "document_search_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__search_documents,
        )

        async def __upsert_documents(
            self,
            namespace,
            document_upsert_request,
            x_pinecone_api_version="2026-01.alpha",
            **kwargs,
        ) -> DocumentUpsertResponse:
            """Upsert documents  # noqa: E501

            Upsert flat JSON documents into a namespace. Documents are indexed based on the configured index schema. Vector fields can be user-specified (e.g., `my_vector`) or use the reserved `_values` key. Text fields are indexed based on schema configuration with `full_text_searchable: true`. For guidance and examples, see [Upsert documents](https://docs.pinecone.io/guides/index-data/upsert-data).  # noqa: E501


            Args:
                namespace (str): The namespace to upsert documents into.
                document_upsert_request (DocumentUpsertRequest):
                x_pinecone_api_version (str): Required date-based version header Defaults to "2026-01.alpha", must be one of ["2026-01.alpha"]

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
                DocumentUpsertResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["namespace"] = namespace
            kwargs["document_upsert_request"] = document_upsert_request
            return await self.call_with_http_info(**kwargs)

        self.upsert_documents = _AsyncioEndpoint(
            settings={
                "response_type": (DocumentUpsertResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/namespaces/{namespace}/documents/upsert",
                "operation_id": "upsert_documents",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace", "document_upsert_request"],
                "required": ["x_pinecone_api_version", "namespace", "document_upsert_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "namespace": (str,),
                    "document_upsert_request": (DocumentUpsertRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "namespace": "path",
                    "document_upsert_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__upsert_documents,
        )
