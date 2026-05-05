"""Index and IndexStatus response models."""

from __future__ import annotations

from typing import Any, cast

from msgspec import Struct

from pinecone._internal.config import normalize_host
from pinecone.models._mixin import StructDictMixin, _struct_to_dict_recursive


class IndexStatus(StructDictMixin, Struct, kw_only=True):
    """Status of an index.

    Attributes:
        ready: Whether the index is ready to accept requests.
        state: Current state of the index (e.g. ``"Ready"``, ``"Initializing"``).
    """

    ready: bool
    state: str


class ServerlessSpecInfo(StructDictMixin, Struct, kw_only=True):
    """Response-side serverless deployment configuration.

    Attributes:
        cloud: Cloud provider (e.g. ``"aws"``, ``"gcp"``, ``"azure"``).
        region: Cloud region (e.g. ``"us-east-1"``).
        read_capacity: Read capacity configuration (``OnDemand`` or
            ``Dedicated``), or ``None`` if the server response omits it.
            When set, contains a ``"mode"`` key plus mode-specific fields.
        source_collection: Source collection name if the index was
            created from a collection, or ``None``.
        schema: Metadata indexing schema, or ``None`` if all metadata
            fields are indexed (the default).
    """

    cloud: str
    region: str
    read_capacity: dict[str, Any] | None = None
    source_collection: str | None = None
    schema: dict[str, Any] | None = None


class PodSpecInfo(StructDictMixin, Struct, kw_only=True):
    """Response-side pod deployment configuration.

    Attributes:
        environment: Deployment environment (e.g. ``"us-east1-gcp"``).
        pod_type: Pod type (e.g. ``"p1.x1"``).
        replicas: Number of replicas.
        shards: Number of shards.
        pods: Total number of pods.
        metadata_config: Metadata indexing configuration, or ``None``.
        source_collection: Source collection name, or ``None``.
    """

    environment: str
    pod_type: str
    replicas: int
    shards: int
    pods: int
    metadata_config: dict[str, list[str]] | None = None
    source_collection: str | None = None


class ByocSpecInfo(StructDictMixin, Struct, kw_only=True):
    """Response-side BYOC (bring your own cloud) deployment configuration.

    Attributes:
        environment: BYOC environment identifier.
        read_capacity: Read capacity configuration, or ``None``.
    """

    environment: str
    read_capacity: dict[str, Any] | None = None


class IndexSpec(StructDictMixin, Struct, kw_only=True):
    """Deployment specification for an index.

    Exactly one of ``serverless``, ``pod``, or ``byoc`` will be set.

    Attributes:
        serverless: Serverless deployment config, or ``None``.
        pod: Pod-based deployment config, or ``None``.
        byoc: BYOC deployment config, or ``None``.
    """

    serverless: ServerlessSpecInfo | None = None
    pod: PodSpecInfo | None = None
    byoc: ByocSpecInfo | None = None


class ModelIndexEmbed(StructDictMixin, Struct, kw_only=True):
    """Embedding configuration for a model-backed (integrated) index.

    Attributes:
        model (str): The name of the embedding model used by this index.
        metric (str | None): Distance metric, or ``None`` if inferred from the model.
        dimension (int | None): Vector dimension, or ``None`` if inferred from the model.
        vector_type (str | None): Vector type (``"dense"`` or ``"sparse"``), or ``None``.
        field_map (dict[str, str] | None): Mapping of document field names to embedding
            input roles, or ``None``.
        read_parameters (dict[str, Any] | None): Model-specific parameters for read
            (query) operations, or ``None``.
        write_parameters (dict[str, Any] | None): Model-specific parameters for write
            (upsert) operations, or ``None``.
    """

    model: str
    metric: str | None = None
    dimension: int | None = None
    vector_type: str | None = None
    field_map: dict[str, str] | None = None
    read_parameters: dict[str, Any] | None = None
    write_parameters: dict[str, Any] | None = None


class IndexTags(dict):  # type: ignore[type-arg]
    """A dict subclass for index tags that adds a ``to_dict()`` helper.

    Backwards-compatible with legacy SDK code that called ``.tags.to_dict()``.
    """

    def to_dict(self) -> dict[str, str]:
        return dict(self)


class IndexModel(Struct, kw_only=True):
    """Response model for a Pinecone index.

    Attributes:
        name: The name of the index.
        metric: Distance metric used for similarity search (e.g. ``"cosine"``,
            ``"euclidean"``, ``"dotproduct"``).
        host: The hostname where this index is served, or ``None`` if the index
            is still initializing and has not yet been assigned a host.
        status: Current status of the index.
        spec: Deployment specification containing either ``serverless``,
            ``pod``, or ``byoc`` configuration.
        vector_type: Type of vectors stored (default: ``"dense"``).
        dimension: Dimensionality of vectors in the index, or ``None`` for
            indexes that infer dimension from the first upsert.
        deletion_protection: Whether deletion protection is enabled
            (``"enabled"`` or ``"disabled"``).
        tags: User-defined key-value tags attached to the index, or ``None``
            if no tags are set.
        embed: Embedding configuration for model-backed (integrated) indexes,
            populated for indexes created with integrated inference and ``None``
            otherwise. See :class:`ModelIndexEmbed`.
        created_at: ISO-8601 timestamp of when the index was created, or
            ``None`` if the server response did not include it. Stored as
            a string; parse with ``datetime.fromisoformat`` if you need
            a ``datetime`` object.
    """

    name: str
    metric: str
    status: IndexStatus
    spec: IndexSpec
    host: str | None = None
    vector_type: str = "dense"
    dimension: int | None = None
    deletion_protection: str = "disabled"
    tags: dict[str, str] | None = None
    embed: ModelIndexEmbed | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize host to always include https:// scheme."""
        if self.host is not None:
            self.host = normalize_host(self.host)
        if isinstance(self.tags, dict) and not isinstance(self.tags, IndexTags):
            self.tags = IndexTags(self.tags)

    def __getattr__(self, name: str) -> Any:
        """Raise AttributeError for unknown attributes (legacy dict-style delegation)."""
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. index['name'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'name' in index``)."""
        return key in self.__struct_fields__

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation, recursively converting nested fields.

        Returns:
            Dictionary with all top-level fields, where nested ``spec``, ``status``,
            and ``embed`` structs are also converted to plain dicts recursively.
            Optional fields (``dimension``, ``tags``, ``embed``) that are ``None``
            are included in the output with their ``None`` values.

        Examples:
            >>> from pinecone.models.indexes.index import (
            ...     IndexModel, IndexSpec, IndexStatus, ServerlessSpecInfo
            ... )
            >>> index = IndexModel(
            ...     name="my-index",
            ...     metric="cosine",
            ...     host="my-index-xyz.svc.pinecone.io",
            ...     status=IndexStatus(ready=True, state="Ready"),
            ...     spec=IndexSpec(serverless=ServerlessSpecInfo(cloud="aws", region="us-east-1")),
            ... )
            >>> d = index.to_dict()
            >>> d["name"]
            'my-index'
            >>> type(d["spec"])
            <class 'dict'>
            >>> d["spec"]["serverless"]
            {'cloud': 'aws', 'region': 'us-east-1'}
        """
        return cast(dict[str, Any], _struct_to_dict_recursive(self))
