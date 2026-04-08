"""Project response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from msgspec import Struct


class ProjectModel(Struct, kw_only=True):
    """Response model for a Pinecone project.

    Attributes:
        id: Unique identifier for the project.
        name: Name of the project.
        max_pods: Maximum number of pods allowed in the project.
        force_encryption_with_cmek: Whether CMEK encryption is enforced.
        organization_id: Identifier of the parent organization.
        created_at: Timestamp when the project was created.
    """

    id: str
    name: str
    max_pods: int
    force_encryption_with_cmek: bool
    organization_id: str
    created_at: str | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. project['name'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'name' in project``)."""
        return key in self.__struct_fields__


class ProjectList:
    """Wrapper around a list of ProjectModel with convenience methods."""

    def __init__(self, projects: list[ProjectModel]) -> None:
        """Initialize a ProjectList.

        Args:
            projects: List of :class:`ProjectModel` instances representing
                Pinecone projects.
        """
        self._projects = projects

    def __iter__(self) -> Iterator[ProjectModel]:
        return iter(self._projects)

    def __len__(self) -> int:
        return len(self._projects)

    def __getitem__(self, index: int) -> ProjectModel:
        return self._projects[index]

    def names(self) -> list[str]:
        """Return a list of project names.

        Returns:
            list[str]: Project names in the same order as the list.

        Examples:
            List names of all projects:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> projects = admin.projects.list()
            >>> projects.names()
            ['production-search', 'staging-recommendations']
        """
        return [project.name for project in self._projects]

    def __repr__(self) -> str:
        summaries = ", ".join(
            f"<name={p.name!r}, id={p.id!r}>" for p in self._projects
        )
        return f"ProjectList([{summaries}])"
