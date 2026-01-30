"""Compatibility shim classes for backward compatibility with the old spec-based API.

These classes map the new deployment-based API structure to the old spec-based
access patterns, ensuring existing code continues to work with the new API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict


@dataclass
class ServerlessSpecCompat:
    """Compatibility shim for serverless spec access.

    Provides the same interface as the old ServerlessSpec response for
    backward compatibility.

    :param cloud: The cloud provider (e.g., "aws", "gcp", "azure").
    :param region: The cloud region (e.g., "us-east-1").
    """

    cloud: str
    region: str


@dataclass
class PodSpecCompat:
    """Compatibility shim for pod spec access.

    Provides the same interface as the old PodSpec response for
    backward compatibility.

    :param environment: The pod environment.
    :param pod_type: The pod type (e.g., "p1.x1").
    :param replicas: Number of replicas.
    :param shards: Number of shards.
    :param pods: Total number of pods.
    :param metadata_config: Metadata configuration dict.
    :param source_collection: Source collection name, if any.
    """

    environment: str
    pod_type: str
    replicas: int
    shards: int
    pods: int
    metadata_config: Dict[str, object] | None = None
    source_collection: str | None = None


@dataclass
class ByocSpecCompat:
    """Compatibility shim for BYOC spec access.

    Provides the same interface as the old ByocSpec response for
    backward compatibility.

    :param environment: The BYOC environment identifier.
    """

    environment: str


class CompatibilitySpec:
    """Compatibility wrapper that provides old-style spec access from new deployment data.

    This class wraps a deployment object from the alpha API and provides the old
    `.spec.serverless` / `.spec.pod` / `.spec.byoc` access patterns for backward
    compatibility.

    :param deployment: The deployment object from the API response.
    """

    def __init__(self, deployment: object):
        self._deployment = deployment

    @property
    def serverless(self) -> ServerlessSpecCompat | None:
        """Get serverless spec if this is a serverless deployment.

        :returns: ServerlessSpecCompat if serverless deployment, None otherwise.
        """
        deployment_type = getattr(self._deployment, "deployment_type", None)
        if deployment_type == "serverless":
            cloud = getattr(self._deployment, "cloud", "")
            region = getattr(self._deployment, "region", "")
            return ServerlessSpecCompat(cloud=cloud, region=region)
        return None

    @property
    def pod(self) -> PodSpecCompat | None:
        """Get pod spec if this is a pod deployment.

        :returns: PodSpecCompat if pod deployment, None otherwise.
        """
        deployment_type = getattr(self._deployment, "deployment_type", None)
        if deployment_type == "pod":
            environment = getattr(self._deployment, "environment", "")
            pod_type = getattr(self._deployment, "pod_type", "p1.x1")
            replicas = getattr(self._deployment, "replicas", 1)
            shards = getattr(self._deployment, "shards", 1)
            pods = getattr(self._deployment, "pods", 1)
            metadata_config = getattr(self._deployment, "metadata_config", None)
            source_collection = getattr(self._deployment, "source_collection", None)
            return PodSpecCompat(
                environment=environment,
                pod_type=pod_type,
                replicas=replicas,
                shards=shards,
                pods=pods,
                metadata_config=metadata_config,
                source_collection=source_collection,
            )
        return None

    @property
    def byoc(self) -> ByocSpecCompat | None:
        """Get BYOC spec if this is a BYOC deployment.

        :returns: ByocSpecCompat if BYOC deployment, None otherwise.
        """
        deployment_type = getattr(self._deployment, "deployment_type", None)
        if deployment_type == "byoc":
            environment = getattr(self._deployment, "environment", "")
            return ByocSpecCompat(environment=environment)
        return None
