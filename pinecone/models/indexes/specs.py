"""Request-side spec structs for index creation."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class EmbedConfig(Struct, frozen=True, kw_only=True):
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
            "model": self.model.value if hasattr(self.model, "value") else self.model,
            "field_map": self.field_map,
        }
        if self.metric is not None:
            result["metric"] = self.metric.value if hasattr(self.metric, "value") else self.metric
        result["read_parameters"] = self.read_parameters if self.read_parameters is not None else {}
        result["write_parameters"] = (
            self.write_parameters if self.write_parameters is not None else {}
        )
        return result


class IntegratedSpec(StructDictMixin, Struct, frozen=True, kw_only=True):  # type: ignore[misc]
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


class ServerlessSpec(StructDictMixin, Struct, frozen=True, kw_only=True, omit_defaults=True):  # type: ignore[misc]
    """Serverless index deployment spec.

    Attributes:
        cloud: Cloud provider (e.g. ``"aws"``, ``"gcp"``, ``"azure"``).
        region: Cloud region (e.g. ``"us-east-1"``, ``"eu-west-1"``).
        read_capacity: Optional read capacity configuration (OnDemand or Dedicated),
            or ``None`` to use the default.
        schema: Optional metadata schema configuration mapping field names to their
            config, or ``None`` for no schema.
    """

    cloud: str
    region: str
    read_capacity: dict[str, Any] | None = None
    schema: dict[str, Any] | None = None

    def asdict(self) -> dict[str, Any]:
        """Return a dict with spec data nested under a ``"serverless"`` key."""
        body: dict[str, Any] = {"cloud": self.cloud, "region": self.region}
        if self.read_capacity is not None:
            body["read_capacity"] = self.read_capacity
        if self.schema is not None:
            body["schema"] = self.schema
        return {"serverless": body}


class PodSpec(StructDictMixin, Struct, frozen=True, kw_only=True):  # type: ignore[misc]
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

    def asdict(self) -> dict[str, Any]:
        """Return a dict with spec data nested under a ``"pod"`` key."""
        body: dict[str, Any] = {
            "environment": self.environment,
            "pod_type": self.pod_type,
            "replicas": self.replicas,
            "shards": self.shards,
            "pods": self.pods,
        }
        if self.metadata_config is not None:
            body["metadata_config"] = self.metadata_config
        if self.source_collection is not None:
            body["source_collection"] = self.source_collection
        return {"pod": body}


class ByocSpec(StructDictMixin, Struct, frozen=True, kw_only=True, omit_defaults=True):  # type: ignore[misc]
    """Bring-your-own-cloud index deployment spec.

    Attributes:
        environment: BYOC environment identifier (e.g. ``"aws-us-east-1-b921"``).
        read_capacity: Optional read capacity configuration (OnDemand or Dedicated).
        schema: Optional metadata schema configuration.
    """

    environment: str
    read_capacity: dict[str, Any] | None = None
    schema: dict[str, Any] | None = None

    def asdict(self) -> dict[str, Any]:
        """Return a dict with spec data nested under a ``"byoc"`` key."""
        body: dict[str, Any] = {"environment": self.environment}
        if self.read_capacity is not None:
            body["read_capacity"] = self.read_capacity
        if self.schema is not None:
            body["schema"] = self.schema
        return {"byoc": body}
