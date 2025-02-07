"""
Pinecone Control Plane API

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
from pinecone.core.openapi.db_control.model.collection_list import CollectionList
from pinecone.core.openapi.db_control.model.collection_model import CollectionModel
from pinecone.core.openapi.db_control.model.configure_index_request import ConfigureIndexRequest
from pinecone.core.openapi.db_control.model.create_collection_request import CreateCollectionRequest
from pinecone.core.openapi.db_control.model.create_index_for_model_request import (
    CreateIndexForModelRequest,
)
from pinecone.core.openapi.db_control.model.create_index_request import CreateIndexRequest
from pinecone.core.openapi.db_control.model.error_response import ErrorResponse
from pinecone.core.openapi.db_control.model.index_list import IndexList
from pinecone.core.openapi.db_control.model.index_model import IndexModel


class ManageIndexesApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __configure_index(
            self, index_name, configure_index_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Configure an index  # noqa: E501

            This operation configures an existing index.   For serverless indexes, you can configure index deletion protection, tags, and integrated inference embedding settings for the index. For pod-based indexes, you can configure the pod size, number of replicas, tags, and index deletion protection.  It is not possible to change the pod type of a pod-based index. However, you can create a collection from a pod-based index and then [create a new pod-based index with a different pod type](http://docs.pinecone.io/guides/indexes/pods/create-a-pod-based-index#create-a-pod-index-from-a-collection) from the collection. For guidance and examples, see [Configure an index](http://docs.pinecone.io/guides/indexes/pods/manage-pod-based-indexes).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.configure_index(index_name, configure_index_request, async_req=True)
            >>> result = thread.get()

            Args:
                index_name (str): The name of the index to configure.
                configure_index_request (ConfigureIndexRequest): The desired pod size and replica configuration for the index.

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
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            kwargs["configure_index_request"] = configure_index_request
            return self.call_with_http_info(**kwargs)

        self.configure_index = _Endpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "configure_index",
                "http_method": "PATCH",
                "servers": None,
            },
            params_map={
                "all": ["index_name", "configure_index_request"],
                "required": ["index_name", "configure_index_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "index_name": (str,),
                    "configure_index_request": (ConfigureIndexRequest,),
                },
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path", "configure_index_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__configure_index,
        )

        def __create_collection(
            self, create_collection_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Create a collection  # noqa: E501

            This operation creates a Pinecone collection.    Serverless indexes do not support collections.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_collection(create_collection_request, async_req=True)
            >>> result = thread.get()

            Args:
                create_collection_request (CreateCollectionRequest): The desired configuration for the collection.

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
                CollectionModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["create_collection_request"] = create_collection_request
            return self.call_with_http_info(**kwargs)

        self.create_collection = _Endpoint(
            settings={
                "response_type": (CollectionModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections",
                "operation_id": "create_collection",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_collection_request"],
                "required": ["create_collection_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_collection_request": (CreateCollectionRequest,)},
                "attribute_map": {},
                "location_map": {"create_collection_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_collection,
        )

        def __create_index(self, create_index_request, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Create an index  # noqa: E501

            This operation deploys a Pinecone index. This is where you specify the measure of similarity, the dimension of vectors to be stored in the index, which cloud provider you would like to deploy with, and more.  For guidance and examples, see [Create an index](https://docs.pinecone.io/guides/indexes/create-an-index#create-a-serverless-index).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_index(create_index_request, async_req=True)
            >>> result = thread.get()

            Args:
                create_index_request (CreateIndexRequest): The desired configuration for the index.

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
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["create_index_request"] = create_index_request
            return self.call_with_http_info(**kwargs)

        self.create_index = _Endpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes",
                "operation_id": "create_index",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_index_request"],
                "required": ["create_index_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_index_request": (CreateIndexRequest,)},
                "attribute_map": {},
                "location_map": {"create_index_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index,
        )

        def __create_index_for_model(
            self, create_index_for_model_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Create an index for an embedding model  # noqa: E501

            This operation creates a serverless integrated inference index for a specific embedding model.  Refer to the [model guide](https://docs.pinecone.io/guides/inference/understanding-inference#embedding-models) for available models and model details.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_index_for_model(create_index_for_model_request, async_req=True)
            >>> result = thread.get()

            Args:
                create_index_for_model_request (CreateIndexForModelRequest): The desired configuration for the index and associated embedding model.

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
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["create_index_for_model_request"] = create_index_for_model_request
            return self.call_with_http_info(**kwargs)

        self.create_index_for_model = _Endpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/create-for-model",
                "operation_id": "create_index_for_model",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_index_for_model_request"],
                "required": ["create_index_for_model_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_index_for_model_request": (CreateIndexForModelRequest,)},
                "attribute_map": {},
                "location_map": {"create_index_for_model_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index_for_model,
        )

        def __delete_collection(self, collection_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Delete a collection  # noqa: E501

            This operation deletes an existing collection. Serverless indexes do not support collections.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_collection(collection_name, async_req=True)
            >>> result = thread.get()

            Args:
                collection_name (str): The name of the collection.

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
            kwargs["collection_name"] = collection_name
            return self.call_with_http_info(**kwargs)

        self.delete_collection = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections/{collection_name}",
                "operation_id": "delete_collection",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["collection_name"],
                "required": ["collection_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"collection_name": (str,)},
                "attribute_map": {"collection_name": "collection_name"},
                "location_map": {"collection_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_collection,
        )

        def __delete_index(self, index_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Delete an index  # noqa: E501

            This operation deletes an existing index.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_index(index_name, async_req=True)
            >>> result = thread.get()

            Args:
                index_name (str): The name of the index to delete.

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
            kwargs["index_name"] = index_name
            return self.call_with_http_info(**kwargs)

        self.delete_index = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "delete_index",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["index_name"],
                "required": ["index_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"index_name": (str,)},
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_index,
        )

        def __describe_collection(self, collection_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Describe a collection  # noqa: E501

            This operation gets a description of a collection. Serverless indexes do not support collections.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_collection(collection_name, async_req=True)
            >>> result = thread.get()

            Args:
                collection_name (str): The name of the collection to be described.

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
                CollectionModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["collection_name"] = collection_name
            return self.call_with_http_info(**kwargs)

        self.describe_collection = _Endpoint(
            settings={
                "response_type": (CollectionModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections/{collection_name}",
                "operation_id": "describe_collection",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["collection_name"],
                "required": ["collection_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"collection_name": (str,)},
                "attribute_map": {"collection_name": "collection_name"},
                "location_map": {"collection_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_collection,
        )

        def __describe_index(self, index_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Describe an index  # noqa: E501

            Get a description of an index.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_index(index_name, async_req=True)
            >>> result = thread.get()

            Args:
                index_name (str): The name of the index to be described.

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
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            return self.call_with_http_info(**kwargs)

        self.describe_index = _Endpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "describe_index",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["index_name"],
                "required": ["index_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"index_name": (str,)},
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_index,
        )

        def __list_collections(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List collections  # noqa: E501

            This operation returns a list of all collections in a project. Serverless indexes do not support collections.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_collections(async_req=True)
            >>> result = thread.get()


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
                CollectionList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_collections = _Endpoint(
            settings={
                "response_type": (CollectionList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections",
                "operation_id": "list_collections",
                "http_method": "GET",
                "servers": None,
            },
            params_map={"all": [], "required": [], "nullable": [], "enum": [], "validation": []},
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {},
                "attribute_map": {},
                "location_map": {},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_collections,
        )

        def __list_indexes(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List indexes  # noqa: E501

            This operation returns a list of all indexes in a project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_indexes(async_req=True)
            >>> result = thread.get()


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
                IndexList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_indexes = _Endpoint(
            settings={
                "response_type": (IndexList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes",
                "operation_id": "list_indexes",
                "http_method": "GET",
                "servers": None,
            },
            params_map={"all": [], "required": [], "nullable": [], "enum": [], "validation": []},
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {},
                "attribute_map": {},
                "location_map": {},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_indexes,
        )


class AsyncioManageIndexesApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __configure_index(self, index_name, configure_index_request, **kwargs):
            """Configure an index  # noqa: E501

            This operation configures an existing index.   For serverless indexes, you can configure index deletion protection, tags, and integrated inference embedding settings for the index. For pod-based indexes, you can configure the pod size, number of replicas, tags, and index deletion protection.  It is not possible to change the pod type of a pod-based index. However, you can create a collection from a pod-based index and then [create a new pod-based index with a different pod type](http://docs.pinecone.io/guides/indexes/pods/create-a-pod-based-index#create-a-pod-index-from-a-collection) from the collection. For guidance and examples, see [Configure an index](http://docs.pinecone.io/guides/indexes/pods/manage-pod-based-indexes).  # noqa: E501


            Args:
                index_name (str): The name of the index to configure.
                configure_index_request (ConfigureIndexRequest): The desired pod size and replica configuration for the index.

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
                IndexModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            kwargs["configure_index_request"] = configure_index_request
            return await self.call_with_http_info(**kwargs)

        self.configure_index = _AsyncioEndpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "configure_index",
                "http_method": "PATCH",
                "servers": None,
            },
            params_map={
                "all": ["index_name", "configure_index_request"],
                "required": ["index_name", "configure_index_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "index_name": (str,),
                    "configure_index_request": (ConfigureIndexRequest,),
                },
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path", "configure_index_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__configure_index,
        )

        async def __create_collection(self, create_collection_request, **kwargs):
            """Create a collection  # noqa: E501

            This operation creates a Pinecone collection.    Serverless indexes do not support collections.   # noqa: E501


            Args:
                create_collection_request (CreateCollectionRequest): The desired configuration for the collection.

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
                CollectionModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["create_collection_request"] = create_collection_request
            return await self.call_with_http_info(**kwargs)

        self.create_collection = _AsyncioEndpoint(
            settings={
                "response_type": (CollectionModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections",
                "operation_id": "create_collection",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_collection_request"],
                "required": ["create_collection_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_collection_request": (CreateCollectionRequest,)},
                "attribute_map": {},
                "location_map": {"create_collection_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_collection,
        )

        async def __create_index(self, create_index_request, **kwargs):
            """Create an index  # noqa: E501

            This operation deploys a Pinecone index. This is where you specify the measure of similarity, the dimension of vectors to be stored in the index, which cloud provider you would like to deploy with, and more.  For guidance and examples, see [Create an index](https://docs.pinecone.io/guides/indexes/create-an-index#create-a-serverless-index).  # noqa: E501


            Args:
                create_index_request (CreateIndexRequest): The desired configuration for the index.

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
                IndexModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["create_index_request"] = create_index_request
            return await self.call_with_http_info(**kwargs)

        self.create_index = _AsyncioEndpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes",
                "operation_id": "create_index",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_index_request"],
                "required": ["create_index_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_index_request": (CreateIndexRequest,)},
                "attribute_map": {},
                "location_map": {"create_index_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index,
        )

        async def __create_index_for_model(self, create_index_for_model_request, **kwargs):
            """Create an index for an embedding model  # noqa: E501

            This operation creates a serverless integrated inference index for a specific embedding model.  Refer to the [model guide](https://docs.pinecone.io/guides/inference/understanding-inference#embedding-models) for available models and model details.  # noqa: E501


            Args:
                create_index_for_model_request (CreateIndexForModelRequest): The desired configuration for the index and associated embedding model.

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
                IndexModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["create_index_for_model_request"] = create_index_for_model_request
            return await self.call_with_http_info(**kwargs)

        self.create_index_for_model = _AsyncioEndpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/create-for-model",
                "operation_id": "create_index_for_model",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_index_for_model_request"],
                "required": ["create_index_for_model_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_index_for_model_request": (CreateIndexForModelRequest,)},
                "attribute_map": {},
                "location_map": {"create_index_for_model_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index_for_model,
        )

        async def __delete_collection(self, collection_name, **kwargs):
            """Delete a collection  # noqa: E501

            This operation deletes an existing collection. Serverless indexes do not support collections.   # noqa: E501


            Args:
                collection_name (str): The name of the collection.

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
            kwargs["collection_name"] = collection_name
            return await self.call_with_http_info(**kwargs)

        self.delete_collection = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections/{collection_name}",
                "operation_id": "delete_collection",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["collection_name"],
                "required": ["collection_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"collection_name": (str,)},
                "attribute_map": {"collection_name": "collection_name"},
                "location_map": {"collection_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_collection,
        )

        async def __delete_index(self, index_name, **kwargs):
            """Delete an index  # noqa: E501

            This operation deletes an existing index.  # noqa: E501


            Args:
                index_name (str): The name of the index to delete.

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
            kwargs["index_name"] = index_name
            return await self.call_with_http_info(**kwargs)

        self.delete_index = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "delete_index",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["index_name"],
                "required": ["index_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"index_name": (str,)},
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_index,
        )

        async def __describe_collection(self, collection_name, **kwargs):
            """Describe a collection  # noqa: E501

            This operation gets a description of a collection. Serverless indexes do not support collections.   # noqa: E501


            Args:
                collection_name (str): The name of the collection to be described.

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
                CollectionModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["collection_name"] = collection_name
            return await self.call_with_http_info(**kwargs)

        self.describe_collection = _AsyncioEndpoint(
            settings={
                "response_type": (CollectionModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections/{collection_name}",
                "operation_id": "describe_collection",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["collection_name"],
                "required": ["collection_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"collection_name": (str,)},
                "attribute_map": {"collection_name": "collection_name"},
                "location_map": {"collection_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_collection,
        )

        async def __describe_index(self, index_name, **kwargs):
            """Describe an index  # noqa: E501

            Get a description of an index.  # noqa: E501


            Args:
                index_name (str): The name of the index to be described.

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
                IndexModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            return await self.call_with_http_info(**kwargs)

        self.describe_index = _AsyncioEndpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "describe_index",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["index_name"],
                "required": ["index_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"index_name": (str,)},
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_index,
        )

        async def __list_collections(self, **kwargs):
            """List collections  # noqa: E501

            This operation returns a list of all collections in a project. Serverless indexes do not support collections.   # noqa: E501



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
                CollectionList
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_collections = _AsyncioEndpoint(
            settings={
                "response_type": (CollectionList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections",
                "operation_id": "list_collections",
                "http_method": "GET",
                "servers": None,
            },
            params_map={"all": [], "required": [], "nullable": [], "enum": [], "validation": []},
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {},
                "attribute_map": {},
                "location_map": {},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_collections,
        )

        async def __list_indexes(self, **kwargs):
            """List indexes  # noqa: E501

            This operation returns a list of all indexes in a project.  # noqa: E501



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
                IndexList
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_indexes = _AsyncioEndpoint(
            settings={
                "response_type": (IndexList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes",
                "operation_id": "list_indexes",
                "http_method": "GET",
                "servers": None,
            },
            params_map={"all": [], "required": [], "nullable": [], "enum": [], "validation": []},
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {},
                "attribute_map": {},
                "location_map": {},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_indexes,
        )
