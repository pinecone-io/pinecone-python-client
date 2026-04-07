"""Namespace response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models.vectors.responses import Pagination


class NamespaceFieldConfig(Struct, kw_only=True):
    """Configuration for a single metadata field in a namespace schema."""

    filterable: bool = False


class NamespaceSchema(Struct, kw_only=True):
    """Schema configuration for a namespace's metadata index."""

    fields: dict[str, NamespaceFieldConfig] = {}


class IndexedFields(Struct, kw_only=True):
    """List of indexed metadata fields in a namespace."""

    fields: list[str] = []


class NamespaceDescription(Struct, kw_only=True):
    """Description of a namespace including name, record count, and schema.

    Attributes:
        name: The name of the namespace.
        record_count: The total number of records in the namespace.
        schema: Schema configuration for metadata indexing, or None.
        indexed_fields: List of indexed metadata fields, or None.
    """

    name: str = ""
    record_count: int = 0
    schema: NamespaceSchema | None = None
    indexed_fields: IndexedFields | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. ns['name'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'name' in ns``)."""
        return key in self.__struct_fields__


class ListNamespacesResponse(Struct, kw_only=True):
    """Response from a list namespaces operation.

    Attributes:
        namespaces: List of namespace descriptions in this page.
        pagination: Pagination token for the next page, or None if last page.
        total_count: Total number of namespaces matching the query.
    """

    namespaces: list[NamespaceDescription] = []
    pagination: Pagination | None = None
    total_count: int = 0

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['namespaces'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'namespaces' in response``)."""
        return key in self.__struct_fields__
