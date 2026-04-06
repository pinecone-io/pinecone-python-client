"""Request-side spec structs for index creation."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class ServerlessSpec(Struct, kw_only=True):
    """Serverless index deployment spec.

    Attributes:
        cloud: Cloud provider (e.g. ``"aws"``, ``"gcp"``, ``"azure"``).
        region: Cloud region (e.g. ``"us-east-1"``, ``"eu-west-1"``).
    """

    cloud: str
    region: str


class PodSpec(Struct, kw_only=True):
    """Pod-based index deployment spec.

    Attributes:
        environment: Deployment environment (e.g. ``"us-east-1-aws"``).
        pod_type: Pod type and size (default: ``"p1.x1"``).
        replicas: Number of replicas (default: 1).
        shards: Number of shards (default: 1).
        pods: Total number of pods (default: 1).
        metadata_config: Configuration for metadata indexing, or ``None``
            to use the default configuration.
        source_collection: Name of a collection to create the index from,
            or ``None`` if creating an empty index.
    """

    environment: str
    pod_type: str = "p1.x1"
    replicas: int = 1
    shards: int = 1
    pods: int = 1
    metadata_config: dict[str, Any] | None = None
    source_collection: str | None = None


class ByocSpec(Struct, kw_only=True):
    """Bring-your-own-cloud index deployment spec.

    Attributes:
        cloud: Cloud provider (e.g. ``"aws"``, ``"gcp"``, ``"azure"``).
        region: Cloud region (e.g. ``"us-east-1"``, ``"eu-west-1"``).
    """

    cloud: str
    region: str
