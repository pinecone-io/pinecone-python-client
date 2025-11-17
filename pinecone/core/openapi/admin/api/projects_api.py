"""
Pinecone Admin API

Provides an API for managing a Pinecone organization and its resources.   # noqa: E501

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
from pinecone.core.openapi.admin.model.create_project_request import CreateProjectRequest
from pinecone.core.openapi.admin.model.error_response import ErrorResponse
from pinecone.core.openapi.admin.model.project import Project
from pinecone.core.openapi.admin.model.project_list import ProjectList
from pinecone.core.openapi.admin.model.update_project_request import UpdateProjectRequest


class ProjectsApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __create_project(
            self,
            create_project_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> Project | ApplyResult[Project]:
            """Create a new project  # noqa: E501

            Creates a new project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_project(create_project_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                create_project_request (CreateProjectRequest): The details of the new project.
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
                Project
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["create_project_request"] = create_project_request
            return cast(Project | ApplyResult[Project], self.call_with_http_info(**kwargs))

        self.create_project = _Endpoint(
            settings={
                "response_type": (Project,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects",
                "operation_id": "create_project",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "create_project_request"],
                "required": ["x_pinecone_api_version", "create_project_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "create_project_request": (CreateProjectRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "create_project_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_project,
        )

        def __delete_project(
            self,
            project_id,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> None:
            """Delete a project  # noqa: E501

            Delete a project and all its associated configuration. Before deleting a project, you must delete all indexes, assistants, backups, and collections associated with the project. Other project resources, such as API keys, are automatically deleted when the project is deleted.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_project(project_id, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                project_id (str): Project ID
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
            kwargs["project_id"] = project_id
            return cast(None, self.call_with_http_info(**kwargs))

        self.delete_project = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}",
                "operation_id": "delete_project",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "project_id"],
                "required": ["x_pinecone_api_version", "project_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "project_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "project_id": "project_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "project_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_project,
        )

        def __fetch_project(
            self,
            project_id,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> Project | ApplyResult[Project]:
            """Get project details  # noqa: E501

            Get details about a project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.fetch_project(project_id, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                project_id (str): Project ID
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
                Project
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["project_id"] = project_id
            return cast(Project | ApplyResult[Project], self.call_with_http_info(**kwargs))

        self.fetch_project = _Endpoint(
            settings={
                "response_type": (Project,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}",
                "operation_id": "fetch_project",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "project_id"],
                "required": ["x_pinecone_api_version", "project_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "project_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "project_id": "project_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "project_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_project,
        )

        def __list_projects(
            self, x_pinecone_api_version="2025-10", **kwargs: ExtraOpenApiKwargsTypedDict
        ) -> ProjectList | ApplyResult[ProjectList]:
            """List projects  # noqa: E501

            List all projects in an organization.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_projects(x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
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
                ProjectList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            return cast(ProjectList | ApplyResult[ProjectList], self.call_with_http_info(**kwargs))

        self.list_projects = _Endpoint(
            settings={
                "response_type": (ProjectList,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects",
                "operation_id": "list_projects",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version"],
                "required": ["x_pinecone_api_version"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,)},
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_projects,
        )

        def __update_project(
            self,
            project_id,
            update_project_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> Project | ApplyResult[Project]:
            """Update a project  # noqa: E501

            Update a project's configuration details. You can update the project's name, maximum number of Pods, or enable encryption with a customer-managed encryption key (CMEK).   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.update_project(project_id, update_project_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                project_id (str): Project ID
                update_project_request (UpdateProjectRequest): Project details to be updated. Fields that are omitted will not be updated.
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
                Project
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["project_id"] = project_id
            kwargs["update_project_request"] = update_project_request
            return cast(Project | ApplyResult[Project], self.call_with_http_info(**kwargs))

        self.update_project = _Endpoint(
            settings={
                "response_type": (Project,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}",
                "operation_id": "update_project",
                "http_method": "PATCH",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "project_id", "update_project_request"],
                "required": ["x_pinecone_api_version", "project_id", "update_project_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "project_id": (str,),
                    "update_project_request": (UpdateProjectRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "project_id": "project_id",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "project_id": "path",
                    "update_project_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__update_project,
        )


class AsyncioProjectsApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __create_project(
            self, create_project_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> Project:
            """Create a new project  # noqa: E501

            Creates a new project.  # noqa: E501


            Args:
                create_project_request (CreateProjectRequest): The details of the new project.
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
                Project
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["create_project_request"] = create_project_request
            return cast(Project, await self.call_with_http_info(**kwargs))

        self.create_project = _AsyncioEndpoint(
            settings={
                "response_type": (Project,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects",
                "operation_id": "create_project",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "create_project_request"],
                "required": ["x_pinecone_api_version", "create_project_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "create_project_request": (CreateProjectRequest,),
                },
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "create_project_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_project,
        )

        async def __delete_project(
            self, project_id, x_pinecone_api_version="2025-10", **kwargs
        ) -> None:
            """Delete a project  # noqa: E501

            Delete a project and all its associated configuration. Before deleting a project, you must delete all indexes, assistants, backups, and collections associated with the project. Other project resources, such as API keys, are automatically deleted when the project is deleted.   # noqa: E501


            Args:
                project_id (str): Project ID
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
            kwargs["project_id"] = project_id
            return cast(None, await self.call_with_http_info(**kwargs))

        self.delete_project = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}",
                "operation_id": "delete_project",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "project_id"],
                "required": ["x_pinecone_api_version", "project_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "project_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "project_id": "project_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "project_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_project,
        )

        async def __fetch_project(
            self, project_id, x_pinecone_api_version="2025-10", **kwargs
        ) -> Project:
            """Get project details  # noqa: E501

            Get details about a project.  # noqa: E501


            Args:
                project_id (str): Project ID
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
                Project
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["project_id"] = project_id
            return cast(Project, await self.call_with_http_info(**kwargs))

        self.fetch_project = _AsyncioEndpoint(
            settings={
                "response_type": (Project,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}",
                "operation_id": "fetch_project",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "project_id"],
                "required": ["x_pinecone_api_version", "project_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "project_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "project_id": "project_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "project_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_project,
        )

        async def __list_projects(self, x_pinecone_api_version="2025-10", **kwargs) -> ProjectList:
            """List projects  # noqa: E501

            List all projects in an organization.  # noqa: E501


            Args:
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
                ProjectList
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            return cast(ProjectList, await self.call_with_http_info(**kwargs))

        self.list_projects = _AsyncioEndpoint(
            settings={
                "response_type": (ProjectList,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects",
                "operation_id": "list_projects",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version"],
                "required": ["x_pinecone_api_version"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,)},
                "attribute_map": {"x_pinecone_api_version": "X-Pinecone-Api-Version"},
                "location_map": {"x_pinecone_api_version": "header"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_projects,
        )

        async def __update_project(
            self, project_id, update_project_request, x_pinecone_api_version="2025-10", **kwargs
        ) -> Project:
            """Update a project  # noqa: E501

            Update a project's configuration details. You can update the project's name, maximum number of Pods, or enable encryption with a customer-managed encryption key (CMEK).   # noqa: E501


            Args:
                project_id (str): Project ID
                update_project_request (UpdateProjectRequest): Project details to be updated. Fields that are omitted will not be updated.
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
                Project
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["project_id"] = project_id
            kwargs["update_project_request"] = update_project_request
            return cast(Project, await self.call_with_http_info(**kwargs))

        self.update_project = _AsyncioEndpoint(
            settings={
                "response_type": (Project,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/projects/{project_id}",
                "operation_id": "update_project",
                "http_method": "PATCH",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "project_id", "update_project_request"],
                "required": ["x_pinecone_api_version", "project_id", "update_project_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "project_id": (str,),
                    "update_project_request": (UpdateProjectRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "project_id": "project_id",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "project_id": "path",
                    "update_project_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__update_project,
        )
