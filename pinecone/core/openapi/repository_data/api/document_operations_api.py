"""
Pinecone Data Plane API for Repositories

Pinecone Repositories build on the vector database to make it easy to store, search and retrieve your data.  # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: unstable
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
from pinecone.core.openapi.repository_data.model.delete_document_response import (
    DeleteDocumentResponse,
)
from pinecone.core.openapi.repository_data.model.get_document_response import GetDocumentResponse
from pinecone.core.openapi.repository_data.model.list_documents_response import (
    ListDocumentsResponse,
)
from pinecone.core.openapi.repository_data.model.upsert_document_response import (
    UpsertDocumentResponse,
)


class DocumentOperationsApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __delete_document(self, namespace, document_id, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Delete a document from the given namespace  # noqa: E501

            Deletes a document from the specified namespace.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_document(namespace, document_id, async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): Namespace to fetch document from.
                document_id (str): Document ID to fetch.

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
                DeleteDocumentResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            kwargs["document_id"] = document_id
            return self.call_with_http_info(**kwargs)

        self.delete_document = _Endpoint(
            settings={
                "response_type": (DeleteDocumentResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/ckb-stub-namespaces/{namespace}/documents/{document_id}",
                "operation_id": "delete_document",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["namespace", "document_id"],
                "required": ["namespace", "document_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,), "document_id": (str,)},
                "attribute_map": {"namespace": "namespace", "document_id": "document_id"},
                "location_map": {"namespace": "path", "document_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_document,
        )

        def __get_document(self, namespace, document_id, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Get a document from the given namespace  # noqa: E501

            Retrieves a document from the specified namespace.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.get_document(namespace, document_id, async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): Namespace to fetch document from.
                document_id (str): Document ID to fetch.

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
                GetDocumentResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            kwargs["document_id"] = document_id
            return self.call_with_http_info(**kwargs)

        self.get_document = _Endpoint(
            settings={
                "response_type": (GetDocumentResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/ckb-stub-namespaces/{namespace}/documents/{document_id}",
                "operation_id": "get_document",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["namespace", "document_id"],
                "required": ["namespace", "document_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,), "document_id": (str,)},
                "attribute_map": {"namespace": "namespace", "document_id": "document_id"},
                "location_map": {"namespace": "path", "document_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__get_document,
        )

        def __list_documents(self, namespace, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List documents from the given namespace  # noqa: E501

            Lists documents from the specified namespace. (Paginated)  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_documents(namespace, async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): Namespace to fetch documents from.

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
                ListDocumentsResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            return self.call_with_http_info(**kwargs)

        self.list_documents = _Endpoint(
            settings={
                "response_type": (ListDocumentsResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/ckb-stub-namespaces/{namespace}/documents",
                "operation_id": "list_documents",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["namespace"],
                "required": ["namespace"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,)},
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_documents,
        )

        def __upsert_document(self, namespace, request_body, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Create or update a document in the given namespace  # noqa: E501

            Upserts a document into the specified namespace.    The request body may contain any valid JSON document that conforms to the schema.    Optionally, an `_id` field can be provided to use as the document's identifier;   if omitted, the system will assign one.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.upsert_document(namespace, request_body, async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): Namespace where the document will be stored.
                request_body ({str: (bool, dict, float, int, list, str, none_type)}):

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
                UpsertDocumentResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            kwargs["request_body"] = request_body
            return self.call_with_http_info(**kwargs)

        self.upsert_document = _Endpoint(
            settings={
                "response_type": (UpsertDocumentResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/ckb-stub-namespaces/{namespace}/documents/upsert",
                "operation_id": "upsert_document",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["namespace", "request_body"],
                "required": ["namespace", "request_body"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "namespace": (str,),
                    "request_body": ({str: (bool, dict, float, int, list, str, none_type)},),
                },
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path", "request_body": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__upsert_document,
        )


class AsyncioDocumentOperationsApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __delete_document(self, namespace, document_id, **kwargs):
            """Delete a document from the given namespace  # noqa: E501

            Deletes a document from the specified namespace.  # noqa: E501


            Args:
                namespace (str): Namespace to fetch document from.
                document_id (str): Document ID to fetch.

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
                DeleteDocumentResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            kwargs["document_id"] = document_id
            return await self.call_with_http_info(**kwargs)

        self.delete_document = _AsyncioEndpoint(
            settings={
                "response_type": (DeleteDocumentResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/ckb-stub-namespaces/{namespace}/documents/{document_id}",
                "operation_id": "delete_document",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["namespace", "document_id"],
                "required": ["namespace", "document_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,), "document_id": (str,)},
                "attribute_map": {"namespace": "namespace", "document_id": "document_id"},
                "location_map": {"namespace": "path", "document_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_document,
        )

        async def __get_document(self, namespace, document_id, **kwargs):
            """Get a document from the given namespace  # noqa: E501

            Retrieves a document from the specified namespace.  # noqa: E501


            Args:
                namespace (str): Namespace to fetch document from.
                document_id (str): Document ID to fetch.

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
                GetDocumentResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            kwargs["document_id"] = document_id
            return await self.call_with_http_info(**kwargs)

        self.get_document = _AsyncioEndpoint(
            settings={
                "response_type": (GetDocumentResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/ckb-stub-namespaces/{namespace}/documents/{document_id}",
                "operation_id": "get_document",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["namespace", "document_id"],
                "required": ["namespace", "document_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,), "document_id": (str,)},
                "attribute_map": {"namespace": "namespace", "document_id": "document_id"},
                "location_map": {"namespace": "path", "document_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__get_document,
        )

        async def __list_documents(self, namespace, **kwargs):
            """List documents from the given namespace  # noqa: E501

            Lists documents from the specified namespace. (Paginated)  # noqa: E501


            Args:
                namespace (str): Namespace to fetch documents from.

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
                ListDocumentsResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            return await self.call_with_http_info(**kwargs)

        self.list_documents = _AsyncioEndpoint(
            settings={
                "response_type": (ListDocumentsResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/ckb-stub-namespaces/{namespace}/documents",
                "operation_id": "list_documents",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["namespace"],
                "required": ["namespace"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"namespace": (str,)},
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_documents,
        )

        async def __upsert_document(self, namespace, request_body, **kwargs):
            """Create or update a document in the given namespace  # noqa: E501

            Upserts a document into the specified namespace.    The request body may contain any valid JSON document that conforms to the schema.    Optionally, an `_id` field can be provided to use as the document's identifier;   if omitted, the system will assign one.  # noqa: E501


            Args:
                namespace (str): Namespace where the document will be stored.
                request_body ({str: (bool, dict, float, int, list, str, none_type)}):

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
                UpsertDocumentResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["namespace"] = namespace
            kwargs["request_body"] = request_body
            return await self.call_with_http_info(**kwargs)

        self.upsert_document = _AsyncioEndpoint(
            settings={
                "response_type": (UpsertDocumentResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/ckb-stub-namespaces/{namespace}/documents/upsert",
                "operation_id": "upsert_document",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["namespace", "request_body"],
                "required": ["namespace", "request_body"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "namespace": (str,),
                    "request_body": ({str: (bool, dict, float, int, list, str, none_type)},),
                },
                "attribute_map": {"namespace": "namespace"},
                "location_map": {"namespace": "path", "request_body": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__upsert_document,
        )
