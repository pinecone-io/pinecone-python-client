"""Adapter for Admin API responses."""

from __future__ import annotations

import msgspec
from msgspec import Struct

from pinecone.models.admin.organization import OrganizationList, OrganizationModel
from pinecone.models.admin.project import ProjectList, ProjectModel


class _OrganizationListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-organizations response."""

    data: list[OrganizationModel] = []


class _ProjectListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-projects response."""

    data: list[ProjectModel] = []


class AdminAdapter:
    """Transforms raw Admin API JSON into domain models."""

    @staticmethod
    def to_organization(data: bytes) -> OrganizationModel:
        """Decode raw JSON bytes into an OrganizationModel."""
        return msgspec.json.decode(data, type=OrganizationModel)

    @staticmethod
    def to_organization_list(data: bytes) -> OrganizationList:
        """Decode raw JSON bytes from a list-organizations response into an OrganizationList."""
        envelope = msgspec.json.decode(data, type=_OrganizationListEnvelope)
        return OrganizationList(envelope.data)

    @staticmethod
    def to_project(data: bytes) -> ProjectModel:
        """Decode raw JSON bytes into a ProjectModel."""
        return msgspec.json.decode(data, type=ProjectModel)

    @staticmethod
    def to_project_list(data: bytes) -> ProjectList:
        """Decode raw JSON bytes from a list-projects response into a ProjectList."""
        envelope = msgspec.json.decode(data, type=_ProjectListEnvelope)
        return ProjectList(envelope.data)
