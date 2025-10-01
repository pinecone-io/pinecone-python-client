"""
    Pinecone Data Plane API for Repositories

    Pinecone Repositories build on the vector database to make it easy to store, search and retrieve your data.  # noqa: E501

    This file is @generated using OpenAPI.

    The version of the OpenAPI document: unstable
    Contact: support@pinecone.io
"""


from pinecone.openapi_support import ApiClient, AsyncioApiClient
from pinecone.openapi_support.endpoint_utils import ExtraOpenApiKwargsTypedDict, KwargsWithOpenApiKwargDefaultsTypedDict
from pinecone.openapi_support.endpoint import Endpoint as _Endpoint, ExtraOpenApiKwargsTypedDict
from pinecone.openapi_support.asyncio_endpoint import AsyncioEndpoint as _AsyncioEndpoint
from pinecone.openapi_support.model_utils import (  # noqa: F401
    date,
    datetime,
    file_type,
    none_type,
    validate_and_convert_types
)
from pinecone.core.openapi.repository_data.model.delete_document_response import DeleteDocumentResponse
from pinecone.core.openapi.repository_data.model.get_document_response import GetDocumentResponse
from pinecone.core.openapi.repository_data.model.list_documents_response import ListDocumentsResponse
from pinecone.core.openapi.repository_data.model.search_documents import SearchDocuments
from pinecone.core.openapi.repository_data.model.search_documents_response import SearchDocumentsResponse
from pinecone.core.openapi.repository_data.model.upsert_document_request import UpsertDocumentRequest
from pinecone.core.openapi.repository_data.model.upsert_document_response import UpsertDocumentResponse


class DocumentOperationsApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __delete_document(
            self,
            namespace,
            document_id,
            x_pinecone_api_version="unstable",
            **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Delete a document from the given namespace  # noqa: E501

            Deletes a document from the specified namespace.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_document(namespace, document_id, x_pinecone_api_version="unstable", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): Namespace to delete the document from.
                document_id (str): Document ID to delete.
                x_pinecone_api_version (str): Required date-based version header Defaults to "unstable", must be one of ["unstable"]

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
            kwargs['x_pinecone_api_version'] = \
                x_pinecone_api_version
            kwargs['namespace'] = \
                namespace
            kwargs['document_id'] = \
                document_id
            return self.call_with_http_info(**kwargs)

        self.delete_document = _Endpoint(
            settings={
                'response_type': (DeleteDocumentResponse,),
                'auth': [
                    'ApiKeyAuth'
                ],
                'endpoint_path': '/namespaces/{namespace}/documents/{document_id}',
                'operation_id': 'delete_document',
                'http_method': 'DELETE',
                'servers': None,
            },
            params_map={
                'all': [
                    'x_pinecone_api_version',
                    'namespace',
                    'document_id',
                ],
                'required': [
                    'x_pinecone_api_version',
                    'namespace',
                    'document_id',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'x_pinecone_api_version':
                        (str,),
                    'namespace':
                        (str,),
                    'document_id':
                        (str,),
                },
                'attribute_map': {
                    'x_pinecone_api_version': 'X-Pinecone-Api-Version',
                    'namespace': 'namespace',
                    'document_id': 'document_id',
                },
                'location_map': {
                    'x_pinecone_api_version': 'header',
                    'namespace': 'path',
                    'document_id': 'path',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client,
            callable=__delete_document
        )

        def __get_document(
            self,
            namespace,
            document_id,
            x_pinecone_api_version="unstable",
            **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Get a document from the given namespace  # noqa: E501

            Retrieves a document from the specified namespace.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.get_document(namespace, document_id, x_pinecone_api_version="unstable", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): Namespace to fetch document from.
                document_id (str): Document ID to fetch.
                x_pinecone_api_version (str): Required date-based version header Defaults to "unstable", must be one of ["unstable"]

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
            kwargs['x_pinecone_api_version'] = \
                x_pinecone_api_version
            kwargs['namespace'] = \
                namespace
            kwargs['document_id'] = \
                document_id
            return self.call_with_http_info(**kwargs)

        self.get_document = _Endpoint(
            settings={
                'response_type': (GetDocumentResponse,),
                'auth': [
                    'ApiKeyAuth'
                ],
                'endpoint_path': '/namespaces/{namespace}/documents/{document_id}',
                'operation_id': 'get_document',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'x_pinecone_api_version',
                    'namespace',
                    'document_id',
                ],
                'required': [
                    'x_pinecone_api_version',
                    'namespace',
                    'document_id',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'x_pinecone_api_version':
                        (str,),
                    'namespace':
                        (str,),
                    'document_id':
                        (str,),
                },
                'attribute_map': {
                    'x_pinecone_api_version': 'X-Pinecone-Api-Version',
                    'namespace': 'namespace',
                    'document_id': 'document_id',
                },
                'location_map': {
                    'x_pinecone_api_version': 'header',
                    'namespace': 'path',
                    'document_id': 'path',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client,
            callable=__get_document
        )

        def __list_documents(
            self,
            namespace,
            x_pinecone_api_version="unstable",
            **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """List documents from the given namespace  # noqa: E501

            Lists documents from the specified namespace. (Paginated)  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_documents(namespace, x_pinecone_api_version="unstable", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): Namespace to fetch documents from.
                x_pinecone_api_version (str): Required date-based version header Defaults to "unstable", must be one of ["unstable"]

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
            kwargs['x_pinecone_api_version'] = \
                x_pinecone_api_version
            kwargs['namespace'] = \
                namespace
            return self.call_with_http_info(**kwargs)

        self.list_documents = _Endpoint(
            settings={
                'response_type': (ListDocumentsResponse,),
                'auth': [
                    'ApiKeyAuth'
                ],
                'endpoint_path': '/namespaces/{namespace}/documents',
                'operation_id': 'list_documents',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'x_pinecone_api_version',
                    'namespace',
                ],
                'required': [
                    'x_pinecone_api_version',
                    'namespace',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'x_pinecone_api_version':
                        (str,),
                    'namespace':
                        (str,),
                },
                'attribute_map': {
                    'x_pinecone_api_version': 'X-Pinecone-Api-Version',
                    'namespace': 'namespace',
                },
                'location_map': {
                    'x_pinecone_api_version': 'header',
                    'namespace': 'path',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client,
            callable=__list_documents
        )

        def __search_documents(
            self,
            namespace,
            search_documents,
            x_pinecone_api_version="unstable",
            **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Search the documents in the given namespace.  # noqa: E501

            Search the documents in the specified namespace.    # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.search_documents(namespace, search_documents, x_pinecone_api_version="unstable", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): Namespace with the documents to search.
                search_documents (SearchDocuments): The configuration for performing a search.
                x_pinecone_api_version (str): Required date-based version header Defaults to "unstable", must be one of ["unstable"]

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
                SearchDocumentsResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs['x_pinecone_api_version'] = \
                x_pinecone_api_version
            kwargs['namespace'] = \
                namespace
            kwargs['search_documents'] = \
                search_documents
            return self.call_with_http_info(**kwargs)

        self.search_documents = _Endpoint(
            settings={
                'response_type': (SearchDocumentsResponse,),
                'auth': [
                    'ApiKeyAuth'
                ],
                'endpoint_path': '/namespaces/{namespace}/documents/search',
                'operation_id': 'search_documents',
                'http_method': 'POST',
                'servers': None,
            },
            params_map={
                'all': [
                    'x_pinecone_api_version',
                    'namespace',
                    'search_documents',
                ],
                'required': [
                    'x_pinecone_api_version',
                    'namespace',
                    'search_documents',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'x_pinecone_api_version':
                        (str,),
                    'namespace':
                        (str,),
                    'search_documents':
                        (SearchDocuments,),
                },
                'attribute_map': {
                    'x_pinecone_api_version': 'X-Pinecone-Api-Version',
                    'namespace': 'namespace',
                },
                'location_map': {
                    'x_pinecone_api_version': 'header',
                    'namespace': 'path',
                    'search_documents': 'body',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client,
            callable=__search_documents
        )

        def __upsert_document(
            self,
            namespace,
            upsert_document_request,
            x_pinecone_api_version="unstable",
            **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Create or update a document in the given namespace  # noqa: E501

            Upserts a document into the specified namespace.    The request body may contain any valid JSON document that conforms to the schema.    Optionally, an `_id` field can be provided to use as the document's identifier;   if omitted, the system will assign one.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.upsert_document(namespace, upsert_document_request, x_pinecone_api_version="unstable", async_req=True)
            >>> result = thread.get()

            Args:
                namespace (str): Namespace where the document will be stored.
                upsert_document_request (UpsertDocumentRequest):
                x_pinecone_api_version (str): Required date-based version header Defaults to "unstable", must be one of ["unstable"]

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
            kwargs['x_pinecone_api_version'] = \
                x_pinecone_api_version
            kwargs['namespace'] = \
                namespace
            kwargs['upsert_document_request'] = \
                upsert_document_request
            return self.call_with_http_info(**kwargs)

        self.upsert_document = _Endpoint(
            settings={
                'response_type': (UpsertDocumentResponse,),
                'auth': [
                    'ApiKeyAuth'
                ],
                'endpoint_path': '/namespaces/{namespace}/documents/upsert',
                'operation_id': 'upsert_document',
                'http_method': 'POST',
                'servers': None,
            },
            params_map={
                'all': [
                    'x_pinecone_api_version',
                    'namespace',
                    'upsert_document_request',
                ],
                'required': [
                    'x_pinecone_api_version',
                    'namespace',
                    'upsert_document_request',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'x_pinecone_api_version':
                        (str,),
                    'namespace':
                        (str,),
                    'upsert_document_request':
                        (UpsertDocumentRequest,),
                },
                'attribute_map': {
                    'x_pinecone_api_version': 'X-Pinecone-Api-Version',
                    'namespace': 'namespace',
                },
                'location_map': {
                    'x_pinecone_api_version': 'header',
                    'namespace': 'path',
                    'upsert_document_request': 'body',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client,
            callable=__upsert_document
        )



class AsyncioDocumentOperationsApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __delete_document(
            self,
            namespace,
            document_id,
            x_pinecone_api_version="unstable",
            **kwargs
        ):
            """Delete a document from the given namespace  # noqa: E501

            Deletes a document from the specified namespace.  # noqa: E501


            Args:
                namespace (str): Namespace to delete the document from.
                document_id (str): Document ID to delete.
                x_pinecone_api_version (str): Required date-based version header Defaults to "unstable", must be one of ["unstable"]

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
            kwargs['x_pinecone_api_version'] = \
                x_pinecone_api_version
            kwargs['namespace'] = \
                namespace
            kwargs['document_id'] = \
                document_id
            return await self.call_with_http_info(**kwargs)

        self.delete_document = _AsyncioEndpoint(
            settings={
                'response_type': (DeleteDocumentResponse,),
                'auth': [
                    'ApiKeyAuth'
                ],
                'endpoint_path': '/namespaces/{namespace}/documents/{document_id}',
                'operation_id': 'delete_document',
                'http_method': 'DELETE',
                'servers': None,
            },
            params_map={
                'all': [
                    'x_pinecone_api_version',
                    'namespace',
                    'document_id',
                ],
                'required': [
                    'x_pinecone_api_version',
                    'namespace',
                    'document_id',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'x_pinecone_api_version':
                        (str,),
                    'namespace':
                        (str,),
                    'document_id':
                        (str,),
                },
                'attribute_map': {
                    'x_pinecone_api_version': 'X-Pinecone-Api-Version',
                    'namespace': 'namespace',
                    'document_id': 'document_id',
                },
                'location_map': {
                    'x_pinecone_api_version': 'header',
                    'namespace': 'path',
                    'document_id': 'path',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client,
            callable=__delete_document
        )

        async def __get_document(
            self,
            namespace,
            document_id,
            x_pinecone_api_version="unstable",
            **kwargs
        ):
            """Get a document from the given namespace  # noqa: E501

            Retrieves a document from the specified namespace.  # noqa: E501


            Args:
                namespace (str): Namespace to fetch document from.
                document_id (str): Document ID to fetch.
                x_pinecone_api_version (str): Required date-based version header Defaults to "unstable", must be one of ["unstable"]

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
            kwargs['x_pinecone_api_version'] = \
                x_pinecone_api_version
            kwargs['namespace'] = \
                namespace
            kwargs['document_id'] = \
                document_id
            return await self.call_with_http_info(**kwargs)

        self.get_document = _AsyncioEndpoint(
            settings={
                'response_type': (GetDocumentResponse,),
                'auth': [
                    'ApiKeyAuth'
                ],
                'endpoint_path': '/namespaces/{namespace}/documents/{document_id}',
                'operation_id': 'get_document',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'x_pinecone_api_version',
                    'namespace',
                    'document_id',
                ],
                'required': [
                    'x_pinecone_api_version',
                    'namespace',
                    'document_id',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'x_pinecone_api_version':
                        (str,),
                    'namespace':
                        (str,),
                    'document_id':
                        (str,),
                },
                'attribute_map': {
                    'x_pinecone_api_version': 'X-Pinecone-Api-Version',
                    'namespace': 'namespace',
                    'document_id': 'document_id',
                },
                'location_map': {
                    'x_pinecone_api_version': 'header',
                    'namespace': 'path',
                    'document_id': 'path',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client,
            callable=__get_document
        )

        async def __list_documents(
            self,
            namespace,
            x_pinecone_api_version="unstable",
            **kwargs
        ):
            """List documents from the given namespace  # noqa: E501

            Lists documents from the specified namespace. (Paginated)  # noqa: E501


            Args:
                namespace (str): Namespace to fetch documents from.
                x_pinecone_api_version (str): Required date-based version header Defaults to "unstable", must be one of ["unstable"]

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
            kwargs['x_pinecone_api_version'] = \
                x_pinecone_api_version
            kwargs['namespace'] = \
                namespace
            return await self.call_with_http_info(**kwargs)

        self.list_documents = _AsyncioEndpoint(
            settings={
                'response_type': (ListDocumentsResponse,),
                'auth': [
                    'ApiKeyAuth'
                ],
                'endpoint_path': '/namespaces/{namespace}/documents',
                'operation_id': 'list_documents',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'x_pinecone_api_version',
                    'namespace',
                ],
                'required': [
                    'x_pinecone_api_version',
                    'namespace',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'x_pinecone_api_version':
                        (str,),
                    'namespace':
                        (str,),
                },
                'attribute_map': {
                    'x_pinecone_api_version': 'X-Pinecone-Api-Version',
                    'namespace': 'namespace',
                },
                'location_map': {
                    'x_pinecone_api_version': 'header',
                    'namespace': 'path',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client,
            callable=__list_documents
        )

        async def __search_documents(
            self,
            namespace,
            search_documents,
            x_pinecone_api_version="unstable",
            **kwargs
        ):
            """Search the documents in the given namespace.  # noqa: E501

            Search the documents in the specified namespace.    # noqa: E501


            Args:
                namespace (str): Namespace with the documents to search.
                search_documents (SearchDocuments): The configuration for performing a search.
                x_pinecone_api_version (str): Required date-based version header Defaults to "unstable", must be one of ["unstable"]

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
                SearchDocumentsResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs['x_pinecone_api_version'] = \
                x_pinecone_api_version
            kwargs['namespace'] = \
                namespace
            kwargs['search_documents'] = \
                search_documents
            return await self.call_with_http_info(**kwargs)

        self.search_documents = _AsyncioEndpoint(
            settings={
                'response_type': (SearchDocumentsResponse,),
                'auth': [
                    'ApiKeyAuth'
                ],
                'endpoint_path': '/namespaces/{namespace}/documents/search',
                'operation_id': 'search_documents',
                'http_method': 'POST',
                'servers': None,
            },
            params_map={
                'all': [
                    'x_pinecone_api_version',
                    'namespace',
                    'search_documents',
                ],
                'required': [
                    'x_pinecone_api_version',
                    'namespace',
                    'search_documents',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'x_pinecone_api_version':
                        (str,),
                    'namespace':
                        (str,),
                    'search_documents':
                        (SearchDocuments,),
                },
                'attribute_map': {
                    'x_pinecone_api_version': 'X-Pinecone-Api-Version',
                    'namespace': 'namespace',
                },
                'location_map': {
                    'x_pinecone_api_version': 'header',
                    'namespace': 'path',
                    'search_documents': 'body',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client,
            callable=__search_documents
        )

        async def __upsert_document(
            self,
            namespace,
            upsert_document_request,
            x_pinecone_api_version="unstable",
            **kwargs
        ):
            """Create or update a document in the given namespace  # noqa: E501

            Upserts a document into the specified namespace.    The request body may contain any valid JSON document that conforms to the schema.    Optionally, an `_id` field can be provided to use as the document's identifier;   if omitted, the system will assign one.  # noqa: E501


            Args:
                namespace (str): Namespace where the document will be stored.
                upsert_document_request (UpsertDocumentRequest):
                x_pinecone_api_version (str): Required date-based version header Defaults to "unstable", must be one of ["unstable"]

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
            kwargs['x_pinecone_api_version'] = \
                x_pinecone_api_version
            kwargs['namespace'] = \
                namespace
            kwargs['upsert_document_request'] = \
                upsert_document_request
            return await self.call_with_http_info(**kwargs)

        self.upsert_document = _AsyncioEndpoint(
            settings={
                'response_type': (UpsertDocumentResponse,),
                'auth': [
                    'ApiKeyAuth'
                ],
                'endpoint_path': '/namespaces/{namespace}/documents/upsert',
                'operation_id': 'upsert_document',
                'http_method': 'POST',
                'servers': None,
            },
            params_map={
                'all': [
                    'x_pinecone_api_version',
                    'namespace',
                    'upsert_document_request',
                ],
                'required': [
                    'x_pinecone_api_version',
                    'namespace',
                    'upsert_document_request',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'x_pinecone_api_version':
                        (str,),
                    'namespace':
                        (str,),
                    'upsert_document_request':
                        (UpsertDocumentRequest,),
                },
                'attribute_map': {
                    'x_pinecone_api_version': 'X-Pinecone-Api-Version',
                    'namespace': 'namespace',
                },
                'location_map': {
                    'x_pinecone_api_version': 'header',
                    'namespace': 'path',
                    'upsert_document_request': 'body',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client,
            callable=__upsert_document
        )
