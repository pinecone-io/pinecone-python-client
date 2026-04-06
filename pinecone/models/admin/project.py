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
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class ProjectList:
    """Wrapper around a list of ProjectModel with convenience methods."""

    def __init__(self, projects: list[ProjectModel]) -> None:
        self._projects = projects

    def __iter__(self) -> Iterator[ProjectModel]:
        return iter(self._projects)

    def __len__(self) -> int:
        return len(self._projects)

    def __getitem__(self, index: int) -> ProjectModel:
        return self._projects[index]

    def names(self) -> list[str]:
        """Return a list of project names."""
        return [project.name for project in self._projects]

    def __repr__(self) -> str:
        return f"ProjectList(projects={self._projects!r})"
