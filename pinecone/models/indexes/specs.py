"""Request-side spec structs for index creation."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class EmbedConfig(Struct, kw_only=True):
    """Configuration for integrated (model-backed) embedding.

    Attributes:
        model: Name of the embedding model (e.g. ``"multilingual-e5-large"``).
        field_map: Maps document field names to embedding inputs
            (e.g. ``{"text": "my_text_field"}``).
        metric: Similarity metric override, or ``None`` to use the model default.
        read_parameters: Optional read-time model parameters.
        write_parameters: Optional write-time model parameters.
    """

    model: str
    field_map: dict[str, str]
    metric: str | None = None
    read_parameters: dict[str, Any] | None = None
    write_parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Read and write parameters default to empty dicts when not set.
        """
        result: dict[str, Any] = {
            "model": self.model,
            "field_map": self.field_map,
        }
        if self.metric is not None:
            result["metric"] = self.metric
        result["read_parameters"] = self.read_parameters if self.read_parameters is not None else {}
        result["write_parameters"] = self.write_parameters if self.write_parameters is not None else {}
        return result


class IntegratedSpec(Struct, kw_only=True):
    """Integrated (model-backed) index deployment spec.

    Wraps cloud/region and embed config into a single convenience
    object. On the wire the ``embed`` config is sent at the top level
    alongside the serverless spec — serialization handles the split.

    Attributes:
        cloud: Cloud provider (e.g. ``"aws"``, ``"gcp"``, ``"azure"``).
        region: Cloud region (e.g. ``"us-east-1"``).
        embed: Embedding model configuration.
    """

    cloud: str
    region: str
    embed: EmbedConfig


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
