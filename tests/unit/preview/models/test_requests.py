"""Unit tests for PreviewCreateIndexRequest and PreviewConfigureIndexRequest."""

from __future__ import annotations

import msgspec

from pinecone.preview.models.deployment import (
    PreviewByocDeployment,
    PreviewManagedDeployment,
    PreviewPodDeployment,
)
from pinecone.preview.models.read_capacity import (
    PreviewReadCapacityDedicatedInner,
    PreviewReadCapacityDedicatedResponse,
    PreviewReadCapacityManualScaling,
    PreviewReadCapacityOnDemandResponse,
    PreviewReadCapacityStatus,
)
from pinecone.preview.models.requests import (
    PreviewConfigureIndexRequest,
    PreviewCreateIndexRequest,
)
from pinecone.preview.models.schema import (
    PreviewDenseVectorField,
    PreviewFullTextSearchConfig,
    PreviewSchema,
    PreviewStringField,
)


def _make_schema(num_fields: int = 1) -> PreviewSchema:
    fields: dict[str, object] = {"vec": PreviewDenseVectorField(dimension=1536, metric="cosine")}
    if num_fields > 1:
        fields["text"] = PreviewStringField(full_text_search=PreviewFullTextSearchConfig())
    return PreviewSchema(fields=fields)  # type: ignore[arg-type]


def test_create_request_with_schema_only() -> None:
    req = PreviewCreateIndexRequest(schema=_make_schema())  # type: ignore[arg-type]
    assert req.name is None
    assert req.deployment is None
    assert req.read_capacity is None
    assert req.deletion_protection is None
    assert req.tags is None


def test_create_request_full() -> None:
    deployment = PreviewManagedDeployment(environment="aped-1", cloud="aws", region="us-east-1")
    rc = PreviewReadCapacityOnDemandResponse(status=PreviewReadCapacityStatus(state="Ready"))
    req = PreviewCreateIndexRequest(
        schema=_make_schema(),  # type: ignore[arg-type]
        name="my-index",
        deployment=deployment,  # type: ignore[arg-type]
        read_capacity=rc,  # type: ignore[arg-type]
        deletion_protection="enabled",
        tags={"env": "prod"},
    )
    result = msgspec.to_builtins(req)
    assert result["name"] == "my-index"
    assert result["deletion_protection"] == "enabled"
    assert result["tags"] == {"env": "prod"}
    assert result["deployment"]["deployment_type"] == "managed"
    assert result["deployment"]["cloud"] == "aws"
    assert result["deployment"]["region"] == "us-east-1"
    assert result["deployment"]["environment"] == "aped-1"
    assert result["read_capacity"]["mode"] == "OnDemand"
    assert result["read_capacity"]["status"]["state"] == "Ready"


def test_create_request_serialization_omits_none() -> None:
    req = PreviewCreateIndexRequest(schema=_make_schema(), name="idx")  # type: ignore[arg-type]
    result = msgspec.to_builtins(req)
    assert set(result.keys()) == {"schema", "name"}


def test_configure_request_all_optional() -> None:
    req = PreviewConfigureIndexRequest()
    result = msgspec.to_builtins(req)
    assert result == {}


def test_configure_request_partial_tags_only() -> None:
    req = PreviewConfigureIndexRequest(tags={"env": "staging"})
    result = msgspec.to_builtins(req)
    assert result == {"tags": {"env": "staging"}}


def test_configure_request_schema_update() -> None:
    schema = _make_schema(num_fields=2)
    req = PreviewConfigureIndexRequest(schema=schema)  # type: ignore[arg-type]
    result = msgspec.to_builtins(req)
    assert "schema" in result
    fields = result["schema"]["fields"]
    assert len(fields) == 2
    # Dense vector field has type discriminator "dense_vector"
    assert fields["vec"]["type"] == "dense_vector"
    # String field has type discriminator "string"
    assert fields["text"]["type"] == "string"


def test_create_request_encodes_pod_deployment() -> None:
    deployment = PreviewPodDeployment(
        environment="us-east1-gcp",
        pod_type="p1.x1",
        pods=2,
        replicas=1,
        shards=1,
    )
    req = PreviewCreateIndexRequest(schema=_make_schema(), deployment=deployment)  # type: ignore[arg-type]
    result = msgspec.to_builtins(req)
    assert result["deployment"]["deployment_type"] == "pod"
    assert result["deployment"]["pod_type"] == "p1.x1"
    assert result["deployment"]["pods"] == 2
    assert result["deployment"]["replicas"] == 1
    assert result["deployment"]["shards"] == 1
    assert result["deployment"]["environment"] == "us-east1-gcp"


def test_create_request_encodes_byoc_deployment_minimal() -> None:
    deployment = PreviewByocDeployment(environment="my-env")
    req = PreviewCreateIndexRequest(schema=_make_schema(), deployment=deployment)  # type: ignore[arg-type]
    result = msgspec.to_builtins(req)
    assert result["deployment"]["deployment_type"] == "byoc"
    assert result["deployment"]["environment"] == "my-env"
    # cloud and region are None (omit_defaults=True does not apply here since ByocDeployment
    # is its own Struct; to_builtins includes None fields from non-omit_defaults structs)
    assert result["deployment"].get("cloud") is None


def test_create_request_encodes_byoc_deployment_full() -> None:
    deployment = PreviewByocDeployment(
        environment="e1",
        cloud="gcp",
        region="us-east1",
    )
    req = PreviewCreateIndexRequest(schema=_make_schema(), deployment=deployment)  # type: ignore[arg-type]
    result = msgspec.to_builtins(req)
    assert result["deployment"]["deployment_type"] == "byoc"
    assert result["deployment"]["cloud"] == "gcp"
    assert result["deployment"]["region"] == "us-east1"


def test_create_request_encodes_dedicated_read_capacity() -> None:
    rc = PreviewReadCapacityDedicatedResponse(
        dedicated=PreviewReadCapacityDedicatedInner(
            node_type="b1",
            scaling="Manual",
            manual=PreviewReadCapacityManualScaling(shards=2, replicas=1),
        ),
        status=PreviewReadCapacityStatus(state="Ready"),
    )
    req = PreviewCreateIndexRequest(schema=_make_schema(), read_capacity=rc)  # type: ignore[arg-type]
    result = msgspec.to_builtins(req)
    assert result["read_capacity"]["mode"] == "Dedicated"
    assert result["read_capacity"]["dedicated"]["node_type"] == "b1"
    assert result["read_capacity"]["dedicated"]["scaling"] == "Manual"
    assert result["read_capacity"]["dedicated"]["manual"]["shards"] == 2
    assert result["read_capacity"]["dedicated"]["manual"]["replicas"] == 1
    assert result["read_capacity"]["status"]["state"] == "Ready"


def test_configure_request_encodes_pod_deployment_partial_update() -> None:
    req = PreviewConfigureIndexRequest(
        deployment=PreviewPodDeployment(  # type: ignore[arg-type]
            environment="us-east1-gcp",
            pod_type="p2.x1",
            replicas=3,
        ),
    )
    result = msgspec.to_builtins(req)
    assert set(result.keys()) == {"deployment"}
    assert result["deployment"]["deployment_type"] == "pod"
    assert result["deployment"]["pod_type"] == "p2.x1"
    assert result["deployment"]["replicas"] == 3
