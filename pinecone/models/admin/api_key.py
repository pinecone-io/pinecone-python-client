"""API key response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from msgspec import Struct


class APIKeyModel(Struct, kw_only=True):
    """Response model for a Pinecone API key.

    Attributes:
        id: Unique identifier for the API key.
        name: Name of the API key.
        project_id: Identifier of the project the key belongs to.
        roles: List of roles assigned to the key.
    """

    id: str
    name: str
    project_id: str
    roles: list[str]
    description: str | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. api_key['name'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'name' in api_key``)."""
        return key in self.__struct_fields__


class APIKeyWithSecret(Struct, kw_only=True):
    """Response model for an API key with its secret value.

    The secret value is only available at creation time.

    Attributes:
        key: The API key metadata.
        value: The secret API key string.
    """

    key: APIKeyModel
    value: str

    def __repr__(self) -> str:
        masked = f"...{self.value[-4:]}" if len(self.value) >= 4 else "***"
        return f"APIKeyWithSecret(key={self.key!r}, value='{masked}')"

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['value'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'value' in response``)."""
        return key in self.__struct_fields__


class APIKeyList:
    """Wrapper around a list of APIKeyModel with convenience methods."""

    def __init__(self, api_keys: list[APIKeyModel]) -> None:
        """Initialize an APIKeyList.

        Args:
            api_keys: List of :class:`APIKeyModel` instances representing
                Pinecone API keys.
        """
        self._api_keys = api_keys

    def __iter__(self) -> Iterator[APIKeyModel]:
        return iter(self._api_keys)

    def __len__(self) -> int:
        return len(self._api_keys)

    def __getitem__(self, index: int) -> APIKeyModel:
        return self._api_keys[index]

    def names(self) -> list[str]:
        """Return a list of API key names.

        Returns:
            list[str]: API key names in the same order as the list.

        Examples:
            List names of all API keys in a project:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> keys = admin.api_keys.list(project_id="proj-abc123")
            >>> keys.names()
            ['prod-search-key', 'ci-pipeline-key']
        """
        return [api_key.name for api_key in self._api_keys]

    def __repr__(self) -> str:
        return f"APIKeyList(api_keys={self._api_keys!r})"
