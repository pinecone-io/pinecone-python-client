"""Projects namespace — list, create, describe, update, and delete operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.validation import require_non_empty
from pinecone.models.admin.project import ProjectList, ProjectModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)


class Projects:
    """Control-plane operations for Pinecone projects.

    Provides methods to list, create, describe, update, and delete projects.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:

        from pinecone import Admin

        admin = Admin(client_id="my-id", client_secret="my-secret")
        for project in admin.projects.list():
            print(project.name)
    """

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
        self._adapter = AdminAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Projects()"

    def list(self) -> ProjectList:
        """List all projects accessible to the authenticated user.

        Returns:
            A :class:`ProjectList` supporting iteration, len(), and index access.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> admin = Admin(client_id="my-id", client_secret="my-secret")
            >>> for project in admin.projects.list():
            ...     print(project.name)
        """
        logger.info("Listing projects")
        response = self._http.get("/admin/projects")
        result = self._adapter.to_project_list(response.content)
        logger.debug("Listed %d projects", len(result))
        return result

    def create(
        self,
        *,
        name: str,
        max_pods: int | None = None,
        force_encryption_with_cmek: bool | None = None,
    ) -> ProjectModel:
        """Create a new project.

        Args:
            name (str): Name for the new project.
            max_pods (int | None): Maximum number of pods allowed. Omitted if None.
            force_encryption_with_cmek (bool | None): Whether to enforce CMEK encryption.
                Omitted if None.

        Returns:
            A :class:`ProjectModel` with the created project details.

        Raises:
            :exc:`ValidationError`: If *name* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> project = admin.projects.create(name="my-project")
            >>> project.name
            'my-project'
        """
        require_non_empty("name", name)
        body: dict[str, Any] = {"name": name}
        if max_pods is not None:
            body["max_pods"] = max_pods
        if force_encryption_with_cmek is not None:
            body["force_encryption_with_cmek"] = force_encryption_with_cmek
        logger.info("Creating project %r", name)
        response = self._http.post("/admin/projects", json=body)
        result = self._adapter.to_project(response.content)
        logger.debug("Created project %r", result.id)
        return result

    def describe(self, *, project_id: str) -> ProjectModel:
        """Get detailed information about a project.

        Args:
            project_id (str): The identifier of the project.

        Returns:
            A :class:`ProjectModel` with full project details.

        Raises:
            :exc:`ValidationError`: If *project_id* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> project = admin.projects.describe(project_id="proj-abc123")
            >>> project.name
            'my-project'
        """
        require_non_empty("project_id", project_id)
        logger.info("Describing project %r", project_id)
        response = self._http.get(f"/admin/projects/{project_id}")
        result = self._adapter.to_project(response.content)
        logger.debug("Described project %r", project_id)
        return result

    def update(
        self,
        *,
        project_id: str,
        name: str | None = None,
        max_pods: int | None = None,
        force_encryption_with_cmek: bool | None = None,
    ) -> ProjectModel:
        """Update a project's settings.

        Args:
            project_id (str): The identifier of the project to update.
            name (str | None): New name for the project.
            max_pods (int | None): New maximum pod count.
            force_encryption_with_cmek (bool | None): New CMEK enforcement setting.

        Returns:
            A :class:`ProjectModel` with the updated project details.

        Raises:
            :exc:`ValidationError`: If *project_id* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> project = admin.projects.update(
            ...     project_id="proj-abc123", name="new-name"
            ... )
            >>> project.name
            'new-name'
        """
        require_non_empty("project_id", project_id)
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if max_pods is not None:
            body["max_pods"] = max_pods
        if force_encryption_with_cmek is not None:
            body["force_encryption_with_cmek"] = force_encryption_with_cmek
        logger.info("Updating project %r", project_id)
        response = self._http.patch(
            f"/admin/projects/{project_id}",
            json=body,
        )
        result = self._adapter.to_project(response.content)
        logger.debug("Updated project %r", project_id)
        return result

    def delete(self, *, project_id: str) -> None:
        """Delete a project.

        Args:
            project_id (str): The identifier of the project to delete.

        Raises:
            :exc:`ValidationError`: If *project_id* is empty.
            :exc:`ApiError`: If the API returns an error response (e.g. 4xx if project
                still has indexes/collections/backups).

        Examples:
            >>> admin.projects.delete(project_id="proj-abc123")
        """
        require_non_empty("project_id", project_id)
        logger.info("Deleting project %r", project_id)
        self._http.delete(f"/admin/projects/{project_id}")
        logger.debug("Deleted project %r", project_id)
