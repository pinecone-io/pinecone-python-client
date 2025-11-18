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
from pinecone.core.openapi.db_data.model.delete_request import DeleteRequest
from pinecone.core.openapi.db_data.model.describe_index_stats_request import (
    DescribeIndexStatsRequest,
)
from pinecone.core.openapi.db_data.model.fetch_by_metadata_request import FetchByMetadataRequest
from pinecone.core.openapi.db_data.model.fetch_by_metadata_response import FetchByMetadataResponse
from pinecone.core.openapi.db_data.model.fetch_response import FetchResponse
from pinecone.core.openapi.db_data.model.index_description import IndexDescription
from pinecone.core.openapi.db_data.model.list_response import ListResponse
from pinecone.core.openapi.db_data.model.query_request import QueryRequest
from pinecone.core.openapi.db_data.model.query_response import QueryResponse
from pinecone.core.openapi.db_data.model.rpc_status import RpcStatus
from pinecone.core.openapi.db_data.model.search_records_request import SearchRecordsRequest
from pinecone.core.openapi.db_data.model.search_records_response import SearchRecordsResponse
from pinecone.core.openapi.db_data.model.update_request import UpdateRequest
from pinecone.core.openapi.db_data.model.update_response import UpdateResponse
from pinecone.core.openapi.db_data.model.upsert_record import UpsertRecord
from pinecone.core.openapi.db_data.model.upsert_request import UpsertRequest
from pinecone.core.openapi.db_data.model.upsert_response import UpsertResponse


class VectorOperationsApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __delete_vectors(
            self,
            delete_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> Dict[str, Any] | ApplyResult[Dict[str, Any]]:
            """Delete vectors  # noqa: E501

            Delete vectors by id from a single namespace.  For guidance and examples, see [Delete data](https://docs.pinecone.io/guides/manage-data/delete-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_vectors(delete_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                delete_request (DeleteRequest):
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
            kwargs["delete_request"] = delete_request
            return cast(
                Dict[str, Any] | ApplyResult[Dict[str, Any]], self.call_with_http_info(**kwargs)
            )

        self.delete_vectors = _Endpoint(
            settings={
                "response_type": (Dict[str, Any],),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/delete",
                "operation_id": "delete_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "delete_request"],
                "required": ["x_pinecone_api_version", "delete_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "delete_request": (DeleteRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header", "delete_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__delete_vectors,
        )

        def __describe_index_stats(
            self,
            describe_index_stats_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> IndexDescription | ApplyResult[IndexDescription]:
            """Get index stats  # noqa: E501

            Return statistics about the contents of an index, including the vector count per namespace, the number of dimensions, and the index fullness.  Serverless indexes scale automatically as needed, so index fullness is relevant only for pod-based indexes.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_index_stats(describe_index_stats_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                describe_index_stats_request (DescribeIndexStatsRequest):
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
                IndexDescription
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["describe_index_stats_request"] = describe_index_stats_request
            return cast(
                IndexDescription | ApplyResult[IndexDescription], self.call_with_http_info(**kwargs)
            )

        self.describe_index_stats = _Endpoint(
            settings={
                "response_type": (IndexDescription,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/describe_index_stats",
                "operation_id": "describe_index_stats",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "describe_index_stats_request"],
                "required": ["x_pinecone_api_version", "describe_index_stats_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "describe_index_stats_request": (DescribeIndexStatsRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "describe_index_stats_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__describe_index_stats,
        )

        def __fetch_vectors(
            self, ids, x_pinecone_api_version="2025-10", **kwargs: ExtraOpenApiKwargsTypedDict
        ) -> FetchResponse | ApplyResult[FetchResponse]:
            """Fetch vectors  # noqa: E501

            Look up and return vectors by ID from a single namespace. The returned vectors include the vector data and/or metadata.  For guidance and examples, see [Fetch data](https://docs.pinecone.io/guides/manage-data/fetch-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.fetch_vectors(ids, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                ids ([str]): The vector IDs to fetch. Does not accept values containing spaces.
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

            Keyword Args:
                namespace (str): The namespace to fetch vectors from. If not provided, the default namespace is used. [optional]
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
                FetchResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["ids"] = ids
            return cast(
                FetchResponse | ApplyResult[FetchResponse], self.call_with_http_info(**kwargs)
            )

        self.fetch_vectors = _Endpoint(
            settings={
                "response_type": (FetchResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/fetch",
                "operation_id": "fetch_vectors",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "ids", "namespace"],
                "required": ["x_pinecone_api_version", "ids"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "ids": ([str],),
                    "namespace": (str,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "ids": "ids",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "ids": "query",
                    "namespace": "query",
                },
                "collection_format_map": {"ids": "multi"},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_vectors,
        )

        def __fetch_vectors_by_metadata(
            self,
            fetch_by_metadata_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> FetchByMetadataResponse | ApplyResult[FetchByMetadataResponse]:
            """Fetch vectors by metadata  # noqa: E501

            Look up and return vectors by metadata filter from a single namespace. The returned vectors include the vector data and/or metadata. For guidance and examples, see [Fetch data](https://docs.pinecone.io/guides/manage-data/fetch-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.fetch_vectors_by_metadata(fetch_by_metadata_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                fetch_by_metadata_request (FetchByMetadataRequest):
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
                FetchByMetadataResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["fetch_by_metadata_request"] = fetch_by_metadata_request
            return cast(
                FetchByMetadataResponse | ApplyResult[FetchByMetadataResponse],
                self.call_with_http_info(**kwargs),
            )

        self.fetch_vectors_by_metadata = _Endpoint(
            settings={
                "response_type": (FetchByMetadataResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/fetch_by_metadata",
                "operation_id": "fetch_vectors_by_metadata",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "fetch_by_metadata_request"],
                "required": ["x_pinecone_api_version", "fetch_by_metadata_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "fetch_by_metadata_request": (FetchByMetadataRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "fetch_by_metadata_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__fetch_vectors_by_metadata,
        )

        def __list_vectors(
            self, x_pinecone_api_version="2025-10", **kwargs: ExtraOpenApiKwargsTypedDict
        ) -> ListResponse | ApplyResult[ListResponse]:
            """List vector IDs  # noqa: E501

            List the IDs of vectors in a single namespace of a serverless index. An optional prefix can be passed to limit the results to IDs with a common prefix.  Returns up to 100 IDs at a time by default in sorted order (bitwise \"C\" collation). If the `limit` parameter is set, `list` returns up to that number of IDs instead. Whenever there are additional IDs to return, the response also includes a `pagination_token` that you can use to get the next batch of IDs. When the response does not include a `pagination_token`, there are no more IDs to return.  For guidance and examples, see [List record IDs](https://docs.pinecone.io/guides/manage-data/list-record-ids).  **Note:** `list` is supported only for serverless indexes.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_vectors(x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

            Keyword Args:
                prefix (str): The vector IDs to fetch. Does not accept values containing spaces. [optional]
                limit (int): Max number of IDs to return per page. [optional] if omitted the server will use the default value of 100.
                pagination_token (str): Pagination token to continue a previous listing operation. [optional]
                namespace (str): The namespace to list vectors from. If not provided, the default namespace is used. [optional]
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
                ListResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            return cast(
                ListResponse | ApplyResult[ListResponse], self.call_with_http_info(**kwargs)
            )

        self.list_vectors = _Endpoint(
            settings={
                "response_type": (ListResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/list",
                "operation_id": "list_vectors",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": [
                    "x_pinecone_api_version",
                    "prefix",
                    "limit",
                    "pagination_token",
                    "namespace",
                ],
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
                    "prefix": (str,),
                    "limit": (int,),
                    "pagination_token": (str,),
                    "namespace": (str,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "prefix": "prefix",
                    "limit": "limit",
                    "pagination_token": "paginationToken",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "prefix": "query",
                    "limit": "query",
                    "pagination_token": "query",
                    "namespace": "query",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_vectors,
        )

        def __query_vectors(
            self,
            query_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> QueryResponse | ApplyResult[QueryResponse]:
            """Search with a vector  # noqa: E501

            Search a namespace using a query vector. It retrieves the ids of the most similar items in a namespace, along with their similarity scores.  For guidance, examples, and limits, see [Search](https://docs.pinecone.io/guides/search/search-overview).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.query_vectors(query_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                query_request (QueryRequest):
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
                QueryResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["query_request"] = query_request
            return cast(
                QueryResponse | ApplyResult[QueryResponse], self.call_with_http_info(**kwargs)
            )

        self.query_vectors = _Endpoint(
            settings={
                "response_type": (QueryResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/query",
                "operation_id": "query_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "query_request"],
                "required": ["x_pinecone_api_version", "query_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "query_request": (QueryRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header", "query_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__query_vectors,
        )

        def __search_records_namespace(
            self,
            namespace,
            search_records_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> SearchRecordsResponse | ApplyResult[SearchRecordsResponse]:
            """Search with text  # noqa: E501

            Search a namespace with a query text, query vector, or record ID and return the most similar records, along with their similarity scores. Optionally, rerank the initial results based on their relevance to the query.   Searching with text is supported only for indexes with [integrated embedding](https://docs.pinecone.io/guides/index-data/indexing-overview#vector-embedding). Searching with a query vector or record ID is supported for all indexes.   For guidance and examples, see [Search](https://docs.pinecone.io/guides/search/search-overview).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.search_records_namespace(namespace, search_records_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): The namespace to search.
                search_records_request (SearchRecordsRequest):
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
                SearchRecordsResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["namespace"] = namespace
            kwargs["search_records_request"] = search_records_request
            return cast(
                SearchRecordsResponse | ApplyResult[SearchRecordsResponse],
                self.call_with_http_info(**kwargs),
            )

        self.search_records_namespace = _Endpoint(
            settings={
                "response_type": (SearchRecordsResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/records/namespaces/{namespace}/search",
                "operation_id": "search_records_namespace",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace", "search_records_request"],
                "required": ["x_pinecone_api_version", "namespace", "search_records_request"],
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
                    "search_records_request": (SearchRecordsRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "namespace": "path",
                    "search_records_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__search_records_namespace,
        )

        def __update_vector(
            self,
            update_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> UpdateResponse | ApplyResult[UpdateResponse]:
            """Update a vector  # noqa: E501

            Update a vector in a namespace. If a value is included, it will overwrite the previous value. If a `set_metadata` is included, the values of the fields specified in it will be added or overwrite the previous value.  For guidance and examples, see [Update data](https://docs.pinecone.io/guides/manage-data/update-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.update_vector(update_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                update_request (UpdateRequest):
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
                UpdateResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["update_request"] = update_request
            return cast(
                UpdateResponse | ApplyResult[UpdateResponse], self.call_with_http_info(**kwargs)
            )

        self.update_vector = _Endpoint(
            settings={
                "response_type": (UpdateResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/update",
                "operation_id": "update_vector",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "update_request"],
                "required": ["x_pinecone_api_version", "update_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "update_request": (UpdateRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header", "update_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__update_vector,
        )

        def __upsert_records_namespace(
            self,
            namespace,
            upsert_record,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> None:
            """Upsert text  # noqa: E501

            Upsert text into a namespace. Pinecone converts the text to vectors automatically using the hosted embedding model associated with the index.  Upserting text is supported only for [indexes with integrated embedding](https://docs.pinecone.io/reference/api/2025-01/control-plane/create_for_model).  For guidance, examples, and limits, see [Upsert data](https://docs.pinecone.io/guides/index-data/upsert-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.upsert_records_namespace(namespace, upsert_record, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): The namespace to upsert records into.
                upsert_record ([UpsertRecord]):
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
            kwargs["namespace"] = namespace
            kwargs["upsert_record"] = upsert_record
            return cast(None, self.call_with_http_info(**kwargs))

        self.upsert_records_namespace = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/records/namespaces/{namespace}/upsert",
                "operation_id": "upsert_records_namespace",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace", "upsert_record"],
                "required": ["x_pinecone_api_version", "namespace", "upsert_record"],
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
                    "upsert_record": ([UpsertRecord],),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "namespace": "path",
                    "upsert_record": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/x-ndjson"]},
            api_client=api_client,
            callable=__upsert_records_namespace,
        )

        def __upsert_vectors(
            self,
            upsert_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> UpsertResponse | ApplyResult[UpsertResponse]:
            """Upsert vectors  # noqa: E501

            Upsert vectors into a namespace. If a new value is upserted for an existing vector ID, it will overwrite the previous value.  For guidance, examples, and limits, see [Upsert data](https://docs.pinecone.io/guides/index-data/upsert-data).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.upsert_vectors(upsert_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                upsert_request (UpsertRequest):
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
                UpsertResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["upsert_request"] = upsert_request
            return cast(
                UpsertResponse | ApplyResult[UpsertResponse], self.call_with_http_info(**kwargs)
            )

        self.upsert_vectors = _Endpoint(
            settings={
                "response_type": (UpsertResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/upsert",
                "operation_id": "upsert_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "upsert_request"],
                "required": ["x_pinecone_api_version", "upsert_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "upsert_request": (UpsertRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header", "upsert_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__upsert_vectors,
        )


class AsyncioVectorOperationsApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __delete_vectors(
            self, delete_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> Dict[str, Any]:
            """Delete vectors  # noqa: E501

            Delete vectors by id from a single namespace.  For guidance and examples, see [Delete data](https://docs.pinecone.io/guides/manage-data/delete-data).  # noqa: E501


            Args:
                delete_request (DeleteRequest):
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
            kwargs["delete_request"] = delete_request
            return cast(Dict[str, Any], await self.call_with_http_info(**kwargs))

        self.delete_vectors = _AsyncioEndpoint(
            settings={
                "response_type": (Dict[str, Any],),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/delete",
                "operation_id": "delete_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "delete_request"],
                "required": ["x_pinecone_api_version", "delete_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "delete_request": (DeleteRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header", "delete_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__delete_vectors,
        )

        async def __describe_index_stats(
            self, describe_index_stats_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> IndexDescription:
            """Get index stats  # noqa: E501

            Return statistics about the contents of an index, including the vector count per namespace, the number of dimensions, and the index fullness.  Serverless indexes scale automatically as needed, so index fullness is relevant only for pod-based indexes.  # noqa: E501


            Args:
                describe_index_stats_request (DescribeIndexStatsRequest):
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
                IndexDescription
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["describe_index_stats_request"] = describe_index_stats_request
            return cast(IndexDescription, await self.call_with_http_info(**kwargs))

        self.describe_index_stats = _AsyncioEndpoint(
            settings={
                "response_type": (IndexDescription,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/describe_index_stats",
                "operation_id": "describe_index_stats",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "describe_index_stats_request"],
                "required": ["x_pinecone_api_version", "describe_index_stats_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "describe_index_stats_request": (DescribeIndexStatsRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "describe_index_stats_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__describe_index_stats,
        )

        async def __fetch_vectors(
            self, ids, x_pinecone_api_version="2025-10", **kwargs
        ) -> FetchResponse:
            """Fetch vectors  # noqa: E501

            Look up and return vectors by ID from a single namespace. The returned vectors include the vector data and/or metadata.  For guidance and examples, see [Fetch data](https://docs.pinecone.io/guides/manage-data/fetch-data).  # noqa: E501


            Args:
                ids ([str]): The vector IDs to fetch. Does not accept values containing spaces.
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

            Keyword Args:
                namespace (str): The namespace to fetch vectors from. If not provided, the default namespace is used. [optional]
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
                FetchResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["ids"] = ids
            return cast(FetchResponse, await self.call_with_http_info(**kwargs))

        self.fetch_vectors = _AsyncioEndpoint(
            settings={
                "response_type": (FetchResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/fetch",
                "operation_id": "fetch_vectors",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "ids", "namespace"],
                "required": ["x_pinecone_api_version", "ids"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "ids": ([str],),
                    "namespace": (str,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "ids": "ids",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "ids": "query",
                    "namespace": "query",
                },
                "collection_format_map": {"ids": "multi"},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_vectors,
        )

        async def __fetch_vectors_by_metadata(
            self, fetch_by_metadata_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> FetchByMetadataResponse:
            """Fetch vectors by metadata  # noqa: E501

            Look up and return vectors by metadata filter from a single namespace. The returned vectors include the vector data and/or metadata. For guidance and examples, see [Fetch data](https://docs.pinecone.io/guides/manage-data/fetch-data).  # noqa: E501


            Args:
                fetch_by_metadata_request (FetchByMetadataRequest):
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
                FetchByMetadataResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["fetch_by_metadata_request"] = fetch_by_metadata_request
            return cast(FetchByMetadataResponse, await self.call_with_http_info(**kwargs))

        self.fetch_vectors_by_metadata = _AsyncioEndpoint(
            settings={
                "response_type": (FetchByMetadataResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/fetch_by_metadata",
                "operation_id": "fetch_vectors_by_metadata",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "fetch_by_metadata_request"],
                "required": ["x_pinecone_api_version", "fetch_by_metadata_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "fetch_by_metadata_request": (FetchByMetadataRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "fetch_by_metadata_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__fetch_vectors_by_metadata,
        )

        async def __list_vectors(self, x_pinecone_api_version="2025-10", **kwargs) -> ListResponse:
            """List vector IDs  # noqa: E501

            List the IDs of vectors in a single namespace of a serverless index. An optional prefix can be passed to limit the results to IDs with a common prefix.  Returns up to 100 IDs at a time by default in sorted order (bitwise \"C\" collation). If the `limit` parameter is set, `list` returns up to that number of IDs instead. Whenever there are additional IDs to return, the response also includes a `pagination_token` that you can use to get the next batch of IDs. When the response does not include a `pagination_token`, there are no more IDs to return.  For guidance and examples, see [List record IDs](https://docs.pinecone.io/guides/manage-data/list-record-ids).  **Note:** `list` is supported only for serverless indexes.  # noqa: E501


            Args:
                x_pinecone_api_version (str): Required date-based version header Defaults to "2025-10", must be one of ["2025-10"]

            Keyword Args:
                prefix (str): The vector IDs to fetch. Does not accept values containing spaces. [optional]
                limit (int): Max number of IDs to return per page. [optional] if omitted the server will use the default value of 100.
                pagination_token (str): Pagination token to continue a previous listing operation. [optional]
                namespace (str): The namespace to list vectors from. If not provided, the default namespace is used. [optional]
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
                ListResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            return cast(ListResponse, await self.call_with_http_info(**kwargs))

        self.list_vectors = _AsyncioEndpoint(
            settings={
                "response_type": (ListResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/list",
                "operation_id": "list_vectors",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": [
                    "x_pinecone_api_version",
                    "prefix",
                    "limit",
                    "pagination_token",
                    "namespace",
                ],
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
                    "prefix": (str,),
                    "limit": (int,),
                    "pagination_token": (str,),
                    "namespace": (str,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "prefix": "prefix",
                    "limit": "limit",
                    "pagination_token": "paginationToken",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "prefix": "query",
                    "limit": "query",
                    "pagination_token": "query",
                    "namespace": "query",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_vectors,
        )

        async def __query_vectors(
            self, query_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> QueryResponse:
            """Search with a vector  # noqa: E501

            Search a namespace using a query vector. It retrieves the ids of the most similar items in a namespace, along with their similarity scores.  For guidance, examples, and limits, see [Search](https://docs.pinecone.io/guides/search/search-overview).  # noqa: E501


            Args:
                query_request (QueryRequest):
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
                QueryResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["query_request"] = query_request
            return cast(QueryResponse, await self.call_with_http_info(**kwargs))

        self.query_vectors = _AsyncioEndpoint(
            settings={
                "response_type": (QueryResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/query",
                "operation_id": "query_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "query_request"],
                "required": ["x_pinecone_api_version", "query_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "query_request": (QueryRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header", "query_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__query_vectors,
        )

        async def __search_records_namespace(
            self, namespace, search_records_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> SearchRecordsResponse:
            """Search with text  # noqa: E501

            Search a namespace with a query text, query vector, or record ID and return the most similar records, along with their similarity scores. Optionally, rerank the initial results based on their relevance to the query.   Searching with text is supported only for indexes with [integrated embedding](https://docs.pinecone.io/guides/index-data/indexing-overview#vector-embedding). Searching with a query vector or record ID is supported for all indexes.   For guidance and examples, see [Search](https://docs.pinecone.io/guides/search/search-overview).  # noqa: E501


            Args:
                namespace (str): The namespace to search.
                search_records_request (SearchRecordsRequest):
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
                SearchRecordsResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["namespace"] = namespace
            kwargs["search_records_request"] = search_records_request
            return cast(SearchRecordsResponse, await self.call_with_http_info(**kwargs))

        self.search_records_namespace = _AsyncioEndpoint(
            settings={
                "response_type": (SearchRecordsResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/records/namespaces/{namespace}/search",
                "operation_id": "search_records_namespace",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace", "search_records_request"],
                "required": ["x_pinecone_api_version", "namespace", "search_records_request"],
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
                    "search_records_request": (SearchRecordsRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "namespace": "path",
                    "search_records_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__search_records_namespace,
        )

        async def __update_vector(
            self, update_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> UpdateResponse:
            """Update a vector  # noqa: E501

            Update a vector in a namespace. If a value is included, it will overwrite the previous value. If a `set_metadata` is included, the values of the fields specified in it will be added or overwrite the previous value.  For guidance and examples, see [Update data](https://docs.pinecone.io/guides/manage-data/update-data).  # noqa: E501


            Args:
                update_request (UpdateRequest):
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
                UpdateResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["update_request"] = update_request
            return cast(UpdateResponse, await self.call_with_http_info(**kwargs))

        self.update_vector = _AsyncioEndpoint(
            settings={
                "response_type": (UpdateResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/update",
                "operation_id": "update_vector",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "update_request"],
                "required": ["x_pinecone_api_version", "update_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "update_request": (UpdateRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header", "update_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__update_vector,
        )

        async def __upsert_records_namespace(
            self, namespace, upsert_record, x_pinecone_api_version="2025-10", **kwargs
        ) -> None:
            """Upsert text  # noqa: E501

            Upsert text into a namespace. Pinecone converts the text to vectors automatically using the hosted embedding model associated with the index.  Upserting text is supported only for [indexes with integrated embedding](https://docs.pinecone.io/reference/api/2025-01/control-plane/create_for_model).  For guidance, examples, and limits, see [Upsert data](https://docs.pinecone.io/guides/index-data/upsert-data).  # noqa: E501


            Args:
                namespace (str): The namespace to upsert records into.
                upsert_record ([UpsertRecord]):
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
            kwargs["namespace"] = namespace
            kwargs["upsert_record"] = upsert_record
            return cast(None, await self.call_with_http_info(**kwargs))

        self.upsert_records_namespace = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/records/namespaces/{namespace}/upsert",
                "operation_id": "upsert_records_namespace",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "namespace", "upsert_record"],
                "required": ["x_pinecone_api_version", "namespace", "upsert_record"],
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
                    "upsert_record": ([UpsertRecord],),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "namespace": "namespace",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "namespace": "path",
                    "upsert_record": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/x-ndjson"]},
            api_client=api_client,
            callable=__upsert_records_namespace,
        )

        async def __upsert_vectors(
            self, upsert_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> UpsertResponse:
            """Upsert vectors  # noqa: E501

            Upsert vectors into a namespace. If a new value is upserted for an existing vector ID, it will overwrite the previous value.  For guidance, examples, and limits, see [Upsert data](https://docs.pinecone.io/guides/index-data/upsert-data).  # noqa: E501


            Args:
                upsert_request (UpsertRequest):
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
                UpsertResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["upsert_request"] = upsert_request
            return cast(UpsertResponse, await self.call_with_http_info(**kwargs))

        self.upsert_vectors = _AsyncioEndpoint(
            settings={
                "response_type": (UpsertResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/vectors/upsert",
                "operation_id": "upsert_vectors",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "upsert_request"],
                "required": ["x_pinecone_api_version", "upsert_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "upsert_request": (UpsertRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header", "upsert_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__upsert_vectors,
        )
