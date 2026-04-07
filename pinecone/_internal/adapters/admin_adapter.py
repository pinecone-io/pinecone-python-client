"""Adapter for Admin API responses."""

from __future__ import annotations

from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
from pinecone.models.admin.api_key import APIKeyList, APIKeyModel, APIKeyWithSecret
from pinecone.models.admin.organization import OrganizationList, OrganizationModel
from pinecone.models.admin.project import ProjectList, ProjectModel


class _OrganizationListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-organizations response."""

    data: list[OrganizationModel] = []


class _APIKeyListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-api-keys response."""

    data: list[APIKeyModel] = []


class _ProjectListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-projects response."""

    data: list[ProjectModel] = []


class AdminAdapter:
    """Transforms raw Admin API JSON into domain models."""

    @staticmethod
    def to_organization(data: bytes) -> OrganizationModel:
        """Decode raw JSON bytes into an OrganizationModel."""
        return decode_response(data, OrganizationModel)

    @staticmethod
    def to_organization_list(data: bytes) -> OrganizationList:
        """Decode raw JSON bytes from a list-organizations response into an OrganizationList."""
        envelope = decode_response(data, _OrganizationListEnvelope)
        return OrganizationList(envelope.data)

    @staticmethod
    def to_project(data: bytes) -> ProjectModel:
        """Decode raw JSON bytes into a ProjectModel."""
        return decode_response(data, ProjectModel)

    @staticmethod
    def to_project_list(data: bytes) -> ProjectList:
        """Decode raw JSON bytes from a list-projects response into a ProjectList."""
        envelope = decode_response(data, _ProjectListEnvelope)
        return ProjectList(envelope.data)

    @staticmethod
    def to_api_key(data: bytes) -> APIKeyModel:
        """Decode raw JSON bytes into an APIKeyModel."""
        return decode_response(data, APIKeyModel)

    @staticmethod
    def to_api_key_with_secret(data: bytes) -> APIKeyWithSecret:
        """Decode raw JSON bytes into an APIKeyWithSecret."""
        return decode_response(data, APIKeyWithSecret)

    @staticmethod
    def to_api_key_list(data: bytes) -> APIKeyList:
        """Decode raw JSON bytes from a list-api-keys response into an APIKeyList."""
        envelope = decode_response(data, _APIKeyListEnvelope)
        return APIKeyList(envelope.data)
