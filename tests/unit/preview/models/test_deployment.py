"""Unit tests for preview deployment models."""

from __future__ import annotations

import msgspec

from pinecone.preview.models.deployment import (
    PreviewByocDeployment,
    PreviewDeployment,
    PreviewManagedDeployment,
    PreviewPodDeployment,
)


def test_managed_deployment_decode() -> None:
    raw = b'{"deployment_type": "managed", "environment": "aped-1", "cloud": "aws", "region": "us-east-1"}'
    result = msgspec.json.decode(raw, type=PreviewDeployment)
    assert isinstance(result, PreviewManagedDeployment)
    assert result.environment == "aped-1"
    assert result.cloud == "aws"
    assert result.region == "us-east-1"


def test_pod_deployment_decode_full() -> None:
    raw = b'{"deployment_type": "pod", "environment": "us-east1-gcp", "pod_type": "p1.x1", "pods": 4, "replicas": 2, "shards": 2}'
    result = msgspec.json.decode(raw, type=PreviewDeployment)
    assert isinstance(result, PreviewPodDeployment)
    assert result.environment == "us-east1-gcp"
    assert result.pod_type == "p1.x1"
    assert result.pods == 4
    assert result.replicas == 2
    assert result.shards == 2


def test_pod_deployment_decode_defaults() -> None:
    raw = b'{"deployment_type": "pod", "environment": "us-east1-gcp", "pod_type": "p1.x1"}'
    result = msgspec.json.decode(raw, type=PreviewDeployment)
    assert isinstance(result, PreviewPodDeployment)
    assert result.pods is None
    assert result.replicas is None
    assert result.shards is None


def test_byoc_deployment_decode_minimal() -> None:
    raw = b'{"deployment_type": "byoc", "environment": "e1"}'
    result = msgspec.json.decode(raw, type=PreviewDeployment)
    assert isinstance(result, PreviewByocDeployment)
    assert result.environment == "e1"
    assert result.cloud is None
    assert result.region is None


def test_byoc_deployment_decode_full() -> None:
    raw = b'{"deployment_type": "byoc", "environment": "e1", "cloud": "gcp", "region": "us-east1"}'
    result = msgspec.json.decode(raw, type=PreviewDeployment)
    assert isinstance(result, PreviewByocDeployment)
    assert result.cloud == "gcp"
    assert result.region == "us-east1"


def test_deployment_union_ignores_unknown_fields() -> None:
    raw = b'{"deployment_type": "managed", "environment": "aped-1", "cloud": "aws", "region": "us-east-1", "metadata_config": {"indexed": ["genre"]}}'
    result = msgspec.json.decode(raw, type=PreviewDeployment)
    assert isinstance(result, PreviewManagedDeployment)
    assert result.environment == "aped-1"
