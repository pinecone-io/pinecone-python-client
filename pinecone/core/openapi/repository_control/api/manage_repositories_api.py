"""
Pinecone Control Plane API for Repositories

Pinecone Repositories make it easy to search and retrieve billions of documents using lexical and semantic search.  # noqa: E501

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
from pinecone.core.openapi.repository_control.model.create_repository_request import (
    CreateRepositoryRequest,
)
from pinecone.core.openapi.repository_control.model.error_response import ErrorResponse
from pinecone.core.openapi.repository_control.model.repository_list import RepositoryList
from pinecone.core.openapi.repository_control.model.repository_model import RepositoryModel


class ManageRepositoriesApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __create_repository(
            self, create_repository_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Create a repository  # noqa: E501

            Create a Pinecone Repository in the cloud provider and region of your choice.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_repository(create_repository_request, async_req=True)
            >>> result = thread.get()

            Args:
                create_repository_request (CreateRepositoryRequest): The desired configuration for the repository.

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
                RepositoryModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["create_repository_request"] = create_repository_request
            return self.call_with_http_info(**kwargs)

        self.create_repository = _Endpoint(
            settings={
                "response_type": (RepositoryModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/repositories",
                "operation_id": "create_repository",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_repository_request"],
                "required": ["create_repository_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_repository_request": (CreateRepositoryRequest,)},
                "attribute_map": {},
                "location_map": {"create_repository_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_repository,
        )

        def __delete_repository(self, repository_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Delete a repository  # noqa: E501

            Delete an existing repository.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_repository(repository_name, async_req=True)
            >>> result = thread.get()

            Args:
                repository_name (str): The name of the repository to delete.

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
            kwargs["repository_name"] = repository_name
            return self.call_with_http_info(**kwargs)

        self.delete_repository = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/repositories/{repository_name}",
                "operation_id": "delete_repository",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["repository_name"],
                "required": ["repository_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"repository_name": (str,)},
                "attribute_map": {"repository_name": "repository_name"},
                "location_map": {"repository_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_repository,
        )

        def __describe_repository(self, repository_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Describe a repository  # noqa: E501

            Get a description of a repository.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_repository(repository_name, async_req=True)
            >>> result = thread.get()

            Args:
                repository_name (str): The name of the repository to be described.

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
                RepositoryModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["repository_name"] = repository_name
            return self.call_with_http_info(**kwargs)

        self.describe_repository = _Endpoint(
            settings={
                "response_type": (RepositoryModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/repositories/{repository_name}",
                "operation_id": "describe_repository",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["repository_name"],
                "required": ["repository_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"repository_name": (str,)},
                "attribute_map": {"repository_name": "repository_name"},
                "location_map": {"repository_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_repository,
        )

        def __list_repositories(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List repositories  # noqa: E501

            List all repositories in a project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_repositories(async_req=True)
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
                RepositoryList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_repositories = _Endpoint(
            settings={
                "response_type": (RepositoryList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/repositories",
                "operation_id": "list_repositories",
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
            callable=__list_repositories,
        )


class AsyncioManageRepositoriesApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __create_repository(self, create_repository_request, **kwargs):
            """Create a repository  # noqa: E501

            Create a Pinecone Repository in the cloud provider and region of your choice.   # noqa: E501


            Args:
                create_repository_request (CreateRepositoryRequest): The desired configuration for the repository.

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
                RepositoryModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["create_repository_request"] = create_repository_request
            return await self.call_with_http_info(**kwargs)

        self.create_repository = _AsyncioEndpoint(
            settings={
                "response_type": (RepositoryModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/repositories",
                "operation_id": "create_repository",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_repository_request"],
                "required": ["create_repository_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_repository_request": (CreateRepositoryRequest,)},
                "attribute_map": {},
                "location_map": {"create_repository_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_repository,
        )

        async def __delete_repository(self, repository_name, **kwargs):
            """Delete a repository  # noqa: E501

            Delete an existing repository.  # noqa: E501


            Args:
                repository_name (str): The name of the repository to delete.

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
            kwargs["repository_name"] = repository_name
            return await self.call_with_http_info(**kwargs)

        self.delete_repository = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/repositories/{repository_name}",
                "operation_id": "delete_repository",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["repository_name"],
                "required": ["repository_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"repository_name": (str,)},
                "attribute_map": {"repository_name": "repository_name"},
                "location_map": {"repository_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_repository,
        )

        async def __describe_repository(self, repository_name, **kwargs):
            """Describe a repository  # noqa: E501

            Get a description of a repository.  # noqa: E501


            Args:
                repository_name (str): The name of the repository to be described.

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
                RepositoryModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["repository_name"] = repository_name
            return await self.call_with_http_info(**kwargs)

        self.describe_repository = _AsyncioEndpoint(
            settings={
                "response_type": (RepositoryModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/repositories/{repository_name}",
                "operation_id": "describe_repository",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["repository_name"],
                "required": ["repository_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"repository_name": (str,)},
                "attribute_map": {"repository_name": "repository_name"},
                "location_map": {"repository_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_repository,
        )

        async def __list_repositories(self, **kwargs):
            """List repositories  # noqa: E501

            List all repositories in a project.  # noqa: E501



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
                RepositoryList
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_repositories = _AsyncioEndpoint(
            settings={
                "response_type": (RepositoryList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/repositories",
                "operation_id": "list_repositories",
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
            callable=__list_repositories,
        )
