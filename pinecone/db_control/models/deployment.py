"""Deployment model classes for index deployment configurations.

These classes represent different deployment configurations for Pinecone indexes,
including serverless, BYOC (bring-your-own-cloud), and pod-based deployments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServerlessDeployment:
    """Serverless deployment configuration.

    Serverless indexes are fully managed by Pinecone and automatically scale
    based on usage.

    :param cloud: The cloud provider ("aws", "gcp", or "azure").
    :param region: The cloud region (e.g., "us-east-1", "us-central1").

    Example usage::

        from pinecone import ServerlessDeployment

        deployment = ServerlessDeployment(cloud="aws", region="us-east-1")

        pc.create_index(
            name="my-index",
            schema=schema,
            deployment=deployment,
        )
    """

    cloud: str
    region: str

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        return {"deployment_type": "serverless", "cloud": self.cloud, "region": self.region}


@dataclass
class ByocDeployment:
    """Bring-your-own-cloud (BYOC) deployment configuration.

    BYOC deployments run in your own cloud infrastructure with a
    Pinecone-managed control plane.

    :param environment: The BYOC environment identifier.

    Example usage::

        from pinecone import ByocDeployment

        deployment = ByocDeployment(environment="aws-us-east-1-b92")

        pc.create_index(
            name="my-index",
            schema=schema,
            deployment=deployment,
        )
    """

    environment: str

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        return {"deployment_type": "byoc", "environment": self.environment}


@dataclass
class PodDeployment:
    """Pod-based deployment configuration.

    Pod deployments provide dedicated compute resources with configurable
    replicas, shards, and pod types.

    :param environment: The pod environment (e.g., "us-east-1-aws").
    :param pod_type: The pod type (e.g., "p1.x1", "s1.x1", "p2.x1").
    :param replicas: Number of replicas (default: 1).
    :param shards: Number of shards (default: 1).
    :param pods: Total number of pods (replicas * shards). If not specified,
        it will be calculated from replicas and shards.

    Example usage::

        from pinecone import PodDeployment

        deployment = PodDeployment(
            environment="us-east-1-aws",
            pod_type="p1.x1",
            replicas=2,
            shards=1,
        )

        pc.create_index(
            name="my-index",
            schema=schema,
            deployment=deployment,
        )
    """

    environment: str
    pod_type: str
    replicas: int = 1
    shards: int = 1
    pods: int | None = None

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        result: dict = {
            "deployment_type": "pod",
            "environment": self.environment,
            "pod_type": self.pod_type,
            "replicas": self.replicas,
            "shards": self.shards,
        }
        if self.pods is not None:
            result["pods"] = self.pods
        return result


# Type alias for any deployment type
Deployment = ServerlessDeployment | ByocDeployment | PodDeployment
