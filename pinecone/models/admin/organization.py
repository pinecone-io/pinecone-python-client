"""Organization response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from msgspec import Struct


class OrganizationModel(Struct, kw_only=True):
    """Response model for a Pinecone organization.

    Attributes:
        id: Unique identifier for the organization.
        name: Name of the organization.
        plan: Plan tier (e.g. Free, Standard, Enterprise, Dedicated).
        payment_status: Current payment status.
        created_at: Timestamp when the organization was created.
        support_tier: Support tier for the organization.
    """

    id: str
    name: str
    plan: str
    payment_status: str
    created_at: str
    support_tier: str

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. org['name'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class OrganizationList:
    """Wrapper around a list of OrganizationModel with convenience methods."""

    def __init__(self, organizations: list[OrganizationModel]) -> None:
        self._organizations = organizations

    def __iter__(self) -> Iterator[OrganizationModel]:
        return iter(self._organizations)

    def __len__(self) -> int:
        return len(self._organizations)

    def __getitem__(self, index: int) -> OrganizationModel:
        return self._organizations[index]

    def names(self) -> list[str]:
        """Return a list of organization names."""
        return [org.name for org in self._organizations]

    def __repr__(self) -> str:
        return f"OrganizationList(organizations={self._organizations!r})"
