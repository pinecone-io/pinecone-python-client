"""Request-side spec structs for index creation."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class ServerlessSpec(Struct, kw_only=True):
    """Serverless index deployment spec."""

    cloud: str
    region: str


class PodSpec(Struct, kw_only=True):
    """Pod-based index deployment spec."""

    environment: str
    pod_type: str = "p1.x1"
    replicas: int = 1
    shards: int = 1
    pods: int = 1
    metadata_config: dict[str, Any] | None = None
    source_collection: str | None = None


class ByocSpec(Struct, kw_only=True):
    """Bring-your-own-cloud index deployment spec."""

    cloud: str
    region: str
