"""
    Pinecone Control Plane API

    Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

    The version of the OpenAPI document: 2024-10
    Contact: support@pinecone.io
    Generated by: https://openapi-generator.tech
"""

import re  # noqa: F401
import sys  # noqa: F401

from pinecone.core_ea.openapi.shared.api_client import ApiClient, Endpoint as _Endpoint
from pinecone.core_ea.openapi.shared.model_utils import (  # noqa: F401
    check_allowed_values,
    check_validations,
    date,
    datetime,
    file_type,
    none_type,
    validate_and_convert_types,
)
from pinecone.core_ea.openapi.db_control.model.collection_list import CollectionList
from pinecone.core_ea.openapi.db_control.model.collection_model import CollectionModel
from pinecone.core_ea.openapi.db_control.model.configure_index_request import ConfigureIndexRequest
from pinecone.core_ea.openapi.db_control.model.create_collection_request import CreateCollectionRequest
from pinecone.core_ea.openapi.db_control.model.create_index_request import CreateIndexRequest
from pinecone.core_ea.openapi.db_control.model.error_response import ErrorResponse
from pinecone.core_ea.openapi.db_control.model.index_list import IndexList
from pinecone.core_ea.openapi.db_control.model.index_model import IndexModel


class ManageIndexesApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __configure_index(self, index_name, configure_index_request, **kwargs):
            """Configure an index  # noqa: E501

            This operation configures an existing index.   For serverless indexes, you can configure only index deletion protection. For pod-based indexes, you can configure the pod size, number of replicas, and index deletion protection.   It is not possible to change the pod type of a pod-based index. However, you can create a collection from a pod-based index and then [create a new pod-based index with a different pod type](http://docs.pinecone.io/guides/indexes/create-an-index#create-an-index-from-a-collection) from the collection. For guidance and examples, see [Configure an index](http://docs.pinecone.io/guides/indexes/configure-an-index).  # noqa: E501
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
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
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
                "all": [
                    "index_name",
                    "configure_index_request",
                ],
                "required": [
                    "index_name",
                    "configure_index_request",
                ],
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
                "attribute_map": {
                    "index_name": "index_name",
                },
                "location_map": {
                    "index_name": "path",
                    "configure_index_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__configure_index,
        )

        def __create_collection(self, create_collection_request, **kwargs):
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
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                CollectionModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
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
                "all": [
                    "create_collection_request",
                ],
                "required": [
                    "create_collection_request",
                ],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "create_collection_request": (CreateCollectionRequest,),
                },
                "attribute_map": {},
                "location_map": {
                    "create_collection_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_collection,
        )

        def __create_index(self, create_index_request, **kwargs):
            """Create an index  # noqa: E501

            This operation deploys a Pinecone index. This is where you specify the measure of similarity, the dimension of vectors to be stored in the index, which cloud provider you would like to deploy with, and more.    For guidance and examples, see [Create an index](https://docs.pinecone.io/guides/indexes/create-an-index#create-a-serverless-index).   # noqa: E501
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
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
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
                "all": [
                    "create_index_request",
                ],
                "required": [
                    "create_index_request",
                ],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "create_index_request": (CreateIndexRequest,),
                },
                "attribute_map": {},
                "location_map": {
                    "create_index_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index,
        )

        def __delete_collection(self, collection_name, **kwargs):
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
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                None
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
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
                "all": [
                    "collection_name",
                ],
                "required": [
                    "collection_name",
                ],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "collection_name": (str,),
                },
                "attribute_map": {
                    "collection_name": "collection_name",
                },
                "location_map": {
                    "collection_name": "path",
                },
                "collection_format_map": {},
            },
            headers_map={
                "accept": ["application/json"],
                "content_type": [],
            },
            api_client=api_client,
            callable=__delete_collection,
        )

        def __delete_index(self, index_name, **kwargs):
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
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                None
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
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
                "all": [
                    "index_name",
                ],
                "required": [
                    "index_name",
                ],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "index_name": (str,),
                },
                "attribute_map": {
                    "index_name": "index_name",
                },
                "location_map": {
                    "index_name": "path",
                },
                "collection_format_map": {},
            },
            headers_map={
                "accept": ["application/json"],
                "content_type": [],
            },
            api_client=api_client,
            callable=__delete_index,
        )

        def __describe_collection(self, collection_name, **kwargs):
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
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                CollectionModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
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
                "all": [
                    "collection_name",
                ],
                "required": [
                    "collection_name",
                ],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "collection_name": (str,),
                },
                "attribute_map": {
                    "collection_name": "collection_name",
                },
                "location_map": {
                    "collection_name": "path",
                },
                "collection_format_map": {},
            },
            headers_map={
                "accept": ["application/json"],
                "content_type": [],
            },
            api_client=api_client,
            callable=__describe_collection,
        )

        def __describe_index(self, index_name, **kwargs):
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
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
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
                "all": [
                    "index_name",
                ],
                "required": [
                    "index_name",
                ],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "index_name": (str,),
                },
                "attribute_map": {
                    "index_name": "index_name",
                },
                "location_map": {
                    "index_name": "path",
                },
                "collection_format_map": {},
            },
            headers_map={
                "accept": ["application/json"],
                "content_type": [],
            },
            api_client=api_client,
            callable=__describe_index,
        )

        def __list_collections(self, **kwargs):
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
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                CollectionList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
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
            headers_map={
                "accept": ["application/json"],
                "content_type": [],
            },
            api_client=api_client,
            callable=__list_collections,
        )

        def __list_indexes(self, **kwargs):
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
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                IndexList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs["async_req"] = kwargs.get("async_req", False)
            kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
            kwargs["_preload_content"] = kwargs.get("_preload_content", True)
            kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
            kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
            kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
            kwargs["_host_index"] = kwargs.get("_host_index")
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
            headers_map={
                "accept": ["application/json"],
                "content_type": [],
            },
            api_client=api_client,
            callable=__list_indexes,
        )
