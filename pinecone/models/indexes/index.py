"""Index and IndexStatus response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class IndexStatus(Struct, kw_only=True):
    """Status of an index.

    Attributes:
        ready: Whether the index is ready to accept requests.
        state: Current state of the index (e.g. ``"Ready"``, ``"Initializing"``).
    """

    ready: bool
    state: str


class ServerlessSpecInfo(Struct, kw_only=True):
    """Response-side serverless deployment configuration.

    Attributes:
        cloud: Cloud provider (e.g. ``"aws"``, ``"gcp"``, ``"azure"``).
        region: Cloud region (e.g. ``"us-east-1"``).
    """

    cloud: str
    region: str


class PodSpecInfo(Struct, kw_only=True):
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


class ByocSpecInfo(Struct, kw_only=True):
    """Response-side BYOC (bring your own cloud) deployment configuration.

    Attributes:
        environment: BYOC environment identifier.
        read_capacity: Read capacity configuration, or ``None``.
    """

    environment: str
    read_capacity: dict[str, Any] | None = None


class IndexSpec(Struct, kw_only=True):
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


class IndexModel(Struct, kw_only=True):
    """Response model for a Pinecone index.

    Attributes:
        name: The name of the index.
        metric: Distance metric used for similarity search (e.g. ``"cosine"``,
            ``"euclidean"``, ``"dotproduct"``).
        host: The hostname where this index is served.
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
    """

    name: str
    metric: str
    host: str
    status: IndexStatus
    spec: IndexSpec
    vector_type: str = "dense"
    dimension: int | None = None
    deletion_protection: str = "disabled"
    tags: dict[str, str] | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. index['name'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'name' in index``)."""
        return key in self.__struct_fields__
