"""Unit tests for to_dict() on index spec and info models."""

from __future__ import annotations

from pinecone.models.indexes.index import (
    ByocSpecInfo,
    IndexSpec,
    IndexStatus,
    ModelIndexEmbed,
    PodSpecInfo,
    ServerlessSpecInfo,
)
from pinecone.models.indexes.specs import (
    EmbedConfig,
    IntegratedSpec,
    PodSpec,
    ServerlessSpec,
)


def test_index_status_to_dict() -> None:
    result = IndexStatus(ready=True, state="Ready").to_dict()
    assert result == {"ready": True, "state": "Ready"}


def test_serverless_spec_info_to_dict() -> None:
    result = ServerlessSpecInfo(cloud="aws", region="us-east-1").to_dict()
    assert result == {"cloud": "aws", "region": "us-east-1"}


def test_pod_spec_info_to_dict_with_optional_none() -> None:
    result = PodSpecInfo(
        environment="us-east1-gcp",
        pod_type="p1.x1",
        replicas=1,
        shards=1,
        pods=1,
        metadata_config=None,
        source_collection=None,
    ).to_dict()
    assert result["metadata_config"] is None
    assert result["source_collection"] is None
    assert result["environment"] == "us-east1-gcp"


def test_byoc_spec_info_to_dict() -> None:
    result = ByocSpecInfo(environment="aws-us-east-1").to_dict()
    assert "environment" in result
    assert result["environment"] == "aws-us-east-1"


def test_index_spec_to_dict_nested_serverless() -> None:
    spec = IndexSpec(serverless=ServerlessSpecInfo(cloud="aws", region="us-east-1"))
    result = spec.to_dict()
    assert isinstance(result["serverless"], dict)
    assert result["serverless"]["cloud"] == "aws"
    assert result["serverless"]["region"] == "us-east-1"
    assert result["pod"] is None
    assert result["byoc"] is None


def test_model_index_embed_to_dict() -> None:
    result = ModelIndexEmbed(model="ml-e5").to_dict()
    assert result["model"] == "ml-e5"
    assert result["metric"] is None
    assert result["dimension"] is None
    assert result["vector_type"] is None
    assert result["field_map"] is None
    assert result["read_parameters"] is None
    assert result["write_parameters"] is None


def test_serverless_spec_to_dict() -> None:
    result = ServerlessSpec(cloud="aws", region="us-east-1").to_dict()
    assert result["cloud"] == "aws"
    assert result["region"] == "us-east-1"
    assert "serverless" not in result


def test_serverless_spec_asdict_still_works() -> None:
    result = ServerlessSpec(cloud="aws", region="us-east-1").asdict()
    assert result == {"serverless": {"cloud": "aws", "region": "us-east-1"}}


def test_pod_spec_to_dict() -> None:
    result = PodSpec(environment="us-east-1-gcp").to_dict()
    assert result["environment"] == "us-east-1-gcp"
    assert result["pod_type"] == "p1.x1"
    assert result["replicas"] == 1
    assert result["shards"] == 1
    assert result["pods"] == 1
    assert result["metadata_config"] is None
    assert result["source_collection"] is None


def test_integrated_spec_to_dict_nested_embed() -> None:
    spec = IntegratedSpec(
        cloud="aws",
        region="us-east-1",
        embed=EmbedConfig(model="multilingual-e5-large", field_map={"text": "my_text"}),
    )
    result = spec.to_dict()
    assert isinstance(result["embed"], dict)
    assert result["embed"]["model"] == "multilingual-e5-large"
    assert result["cloud"] == "aws"


def test_to_dict_is_pure_read() -> None:
    spec = ServerlessSpecInfo(cloud="aws", region="us-east-1")
    first = spec.to_dict()
    first["cloud"] = "mutated"
    second = spec.to_dict()
    assert second["cloud"] == "aws"
