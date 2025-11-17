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
from pinecone.core.openapi.admin.model.error_response import ErrorResponse
from pinecone.core.openapi.admin.model.organization import Organization
from pinecone.core.openapi.admin.model.organization_list import OrganizationList
from pinecone.core.openapi.admin.model.update_organization_request import UpdateOrganizationRequest


class OrganizationsApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __delete_organization(
            self,
            organization_id,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> None:
            """Delete an organization  # noqa: E501

            Delete an organization and all its associated configuration. Before deleting an organization, you must delete all projects (including indexes, assistants, backups, and collections) associated with the organization.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_organization(organization_id, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                organization_id (str): Organization ID
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
            kwargs["organization_id"] = organization_id
            return cast(None, self.call_with_http_info(**kwargs))

        self.delete_organization = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/organizations/{organization_id}",
                "operation_id": "delete_organization",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "organization_id"],
                "required": ["x_pinecone_api_version", "organization_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "organization_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "organization_id": "organization_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "organization_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_organization,
        )

        def __fetch_organization(
            self,
            organization_id,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> Organization | ApplyResult[Organization]:
            """Get organization details  # noqa: E501

            Get details about an organization.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.fetch_organization(organization_id, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                organization_id (str): Organization ID
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
                Organization
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["organization_id"] = organization_id
            return cast(
                Organization | ApplyResult[Organization], self.call_with_http_info(**kwargs)
            )

        self.fetch_organization = _Endpoint(
            settings={
                "response_type": (Organization,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/organizations/{organization_id}",
                "operation_id": "fetch_organization",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "organization_id"],
                "required": ["x_pinecone_api_version", "organization_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "organization_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "organization_id": "organization_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "organization_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_organization,
        )

        def __list_organizations(
            self, x_pinecone_api_version="2025-10", **kwargs: ExtraOpenApiKwargsTypedDict
        ) -> OrganizationList | ApplyResult[OrganizationList]:
            """List organizations  # noqa: E501

            List all organizations associated with an account.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_organizations(x_pinecone_api_version="2025-10", async_req=True)
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
                OrganizationList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            return cast(
                OrganizationList | ApplyResult[OrganizationList], self.call_with_http_info(**kwargs)
            )

        self.list_organizations = _Endpoint(
            settings={
                "response_type": (OrganizationList,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/organizations",
                "operation_id": "list_organizations",
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
            callable=__list_organizations,
        )

        def __update_organization(
            self,
            organization_id,
            update_organization_request,
            x_pinecone_api_version="2025-10",
            **kwargs: ExtraOpenApiKwargsTypedDict,
        ) -> Organization | ApplyResult[Organization]:
            """Update an organization  # noqa: E501

            Update an organization's name.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.update_organization(organization_id, update_organization_request, x_pinecone_api_version="2025-10", async_req=True)
            >>> result = thread.get()

            Args:
                organization_id (str): Organization ID
                update_organization_request (UpdateOrganizationRequest): Organization details to be updated.
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
                Organization
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["organization_id"] = organization_id
            kwargs["update_organization_request"] = update_organization_request
            return cast(
                Organization | ApplyResult[Organization], self.call_with_http_info(**kwargs)
            )

        self.update_organization = _Endpoint(
            settings={
                "response_type": (Organization,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/organizations/{organization_id}",
                "operation_id": "update_organization",
                "http_method": "PATCH",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "organization_id", "update_organization_request"],
                "required": [
                    "x_pinecone_api_version",
                    "organization_id",
                    "update_organization_request",
                ],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "organization_id": (str,),
                    "update_organization_request": (UpdateOrganizationRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "organization_id": "organization_id",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "organization_id": "path",
                    "update_organization_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__update_organization,
        )


class AsyncioOrganizationsApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __delete_organization(
            self, organization_id, x_pinecone_api_version="2025-10", **kwargs
        ) -> None:
            """Delete an organization  # noqa: E501

            Delete an organization and all its associated configuration. Before deleting an organization, you must delete all projects (including indexes, assistants, backups, and collections) associated with the organization.   # noqa: E501


            Args:
                organization_id (str): Organization ID
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
            kwargs["organization_id"] = organization_id
            return cast(None, await self.call_with_http_info(**kwargs))

        self.delete_organization = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/organizations/{organization_id}",
                "operation_id": "delete_organization",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "organization_id"],
                "required": ["x_pinecone_api_version", "organization_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "organization_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "organization_id": "organization_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "organization_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_organization,
        )

        async def __fetch_organization(
            self, organization_id, x_pinecone_api_version="2025-10", **kwargs
        ) -> Organization:
            """Get organization details  # noqa: E501

            Get details about an organization.  # noqa: E501


            Args:
                organization_id (str): Organization ID
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
                Organization
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["organization_id"] = organization_id
            return cast(Organization, await self.call_with_http_info(**kwargs))

        self.fetch_organization = _AsyncioEndpoint(
            settings={
                "response_type": (Organization,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/organizations/{organization_id}",
                "operation_id": "fetch_organization",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "organization_id"],
                "required": ["x_pinecone_api_version", "organization_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"x_pinecone_api_version": (str,), "organization_id": (str,)},
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "organization_id": "organization_id",
                },
                "location_map": {"x_pinecone_api_version": "header", "organization_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__fetch_organization,
        )

        async def __list_organizations(
            self, x_pinecone_api_version="2025-10", **kwargs
        ) -> OrganizationList:
            """List organizations  # noqa: E501

            List all organizations associated with an account.  # noqa: E501


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
                OrganizationList
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            return cast(OrganizationList, await self.call_with_http_info(**kwargs))

        self.list_organizations = _AsyncioEndpoint(
            settings={
                "response_type": (OrganizationList,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/organizations",
                "operation_id": "list_organizations",
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
            callable=__list_organizations,
        )

        async def __update_organization(
            self,
            organization_id,
            update_organization_request,
            x_pinecone_api_version="2025-10",
            **kwargs,
        ) -> Organization:
            """Update an organization  # noqa: E501

            Update an organization's name.   # noqa: E501


            Args:
                organization_id (str): Organization ID
                update_organization_request (UpdateOrganizationRequest): Organization details to be updated.
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
                Organization
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["x_pinecone_api_version"] = x_pinecone_api_version
            kwargs["organization_id"] = organization_id
            kwargs["update_organization_request"] = update_organization_request
            return cast(Organization, await self.call_with_http_info(**kwargs))

        self.update_organization = _AsyncioEndpoint(
            settings={
                "response_type": (Organization,),
                "auth": ["BearerAuth"],
                "endpoint_path": "/admin/organizations/{organization_id}",
                "operation_id": "update_organization",
                "http_method": "PATCH",
                "servers": None,
            },
            params_map={
                "all": ["x_pinecone_api_version", "organization_id", "update_organization_request"],
                "required": [
                    "x_pinecone_api_version",
                    "organization_id",
                    "update_organization_request",
                ],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "x_pinecone_api_version": (str,),
                    "organization_id": (str,),
                    "update_organization_request": (UpdateOrganizationRequest,),
                },
                "attribute_map": {
                    "x_pinecone_api_version": "X-Pinecone-Api-Version",
                    "organization_id": "organization_id",
                },
                "location_map": {
                    "x_pinecone_api_version": "header",
                    "organization_id": "path",
                    "update_organization_request": "body",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__update_organization,
        )
