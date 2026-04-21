"""API key response models for the Admin API."""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from typing import Any

from msgspec import Struct


class APIKeyRole(str, Enum):
    """Roles that can be assigned to a Pinecone API key.

    Possible values: ``PROJECT_EDITOR``, ``PROJECT_VIEWER``,
    ``CONTROL_PLANE_EDITOR``, ``CONTROL_PLANE_VIEWER``,
    ``DATA_PLANE_EDITOR``, ``DATA_PLANE_VIEWER``.

    Examples:
        Create a read-only API key using the enum:

        >>> from pinecone import Admin
        >>> from pinecone.models.admin.api_key import APIKeyRole
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> result = admin.api_keys.create(
        ...     project_id="proj-abc123",
        ...     name="read-only-key",
        ...     roles=[APIKeyRole.DATA_PLANE_VIEWER],
        ... )
        >>> result.key.roles
        [<APIKeyRole.DATA_PLANE_VIEWER: 'DataPlaneViewer'>]

        Update a key to use control-plane access:

        >>> key = admin.api_keys.update(
        ...     api_key_id="key-abc123",
        ...     roles=[APIKeyRole.CONTROL_PLANE_EDITOR],
        ... )
        >>> key.role
        <APIKeyRole.CONTROL_PLANE_EDITOR: 'ControlPlaneEditor'>
    """

    PROJECT_EDITOR = "ProjectEditor"
    PROJECT_VIEWER = "ProjectViewer"
    CONTROL_PLANE_EDITOR = "ControlPlaneEditor"
    CONTROL_PLANE_VIEWER = "ControlPlaneViewer"
    DATA_PLANE_EDITOR = "DataPlaneEditor"
    DATA_PLANE_VIEWER = "DataPlaneViewer"


class APIKeyModel(Struct, kw_only=True):
    """Response model for a Pinecone API key.

    Attributes:
        id (str): Unique identifier for the API key.
        name (str): Name of the API key.
        project_id (str): Identifier of the project the key belongs to.
        roles (list[APIKeyRole]): List of roles assigned to the key
            (see :class:`APIKeyRole`).
        description (str | None): Optional description for the API key.
            ``None`` if no description was set.

    Examples:
        Retrieve an API key and inspect its fields:

        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> key = admin.api_keys.describe(api_key_id="key-abc123")
        >>> key.id
        'key-abc123'
        >>> key.name
        'prod-search-key'
        >>> key.roles
        [<APIKeyRole.DATA_PLANE_EDITOR: 'DataPlaneEditor'>]
        >>> key.description
        'Used by the search service'
    """

    id: str
    name: str
    project_id: str
    roles: list[APIKeyRole]
    description: str | None = None

    @property
    def role(self) -> APIKeyRole:
        """Singular alias for ``roles`` when the key has exactly one role.

        Returns:
            str: The single role assigned to this key.

        Raises:
            :exc:`ValueError`: If the key has no roles or more than one role.

        Examples:
            Access the role of a single-role key:

            >>> key = admin.api_keys.describe(api_key_id="key-abc123")
            >>> key.role
            <APIKeyRole.DATA_PLANE_EDITOR: 'DataPlaneEditor'>

            Keys with multiple roles raise :exc:`ValueError`; use :attr:`roles` instead:

            >>> try:
            ...     key.role
            ... except ValueError as exc:
            ...     print(exc)
            API key has 2 roles; use .roles to access all
        """
        if len(self.roles) == 0:
            raise ValueError("API key has no roles")
        if len(self.roles) > 1:
            raise ValueError(f"API key has {len(self.roles)} roles; use .roles to access all")
        return self.roles[0]

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
        summaries = ", ".join(
            f"<name={k.name!r}, project_id={k.project_id!r}>" for k in self._api_keys
        )
        return f"APIKeyList([{summaries}])"
