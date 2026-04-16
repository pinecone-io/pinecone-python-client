"""Unit tests for PreviewCreateIndexRequest and PreviewConfigureIndexRequest."""

from __future__ import annotations

import msgspec

from pinecone.preview.models.deployment import PreviewManagedDeployment
from pinecone.preview.models.read_capacity import (
    PreviewReadCapacityOnDemandResponse,
    PreviewReadCapacityStatus,
)
from pinecone.preview.models.requests import (
    PreviewConfigureIndexRequest,
    PreviewCreateIndexRequest,
)
from pinecone.preview.models.schema import (
    PreviewDenseVectorField,
    PreviewSchema,
    PreviewStringField,
)


def _make_schema(num_fields: int = 1) -> PreviewSchema:
    fields: dict[str, object] = {"vec": PreviewDenseVectorField(dimension=1536, metric="cosine")}
    if num_fields > 1:
        fields["text"] = PreviewStringField(full_text_searchable=True)
    return PreviewSchema(fields=fields)  # type: ignore[arg-type]


def test_create_request_with_schema_only() -> None:
    req = PreviewCreateIndexRequest(schema=_make_schema())
    assert req.name is None
    assert req.deployment is None
    assert req.read_capacity is None
    assert req.deletion_protection is None
    assert req.tags is None


def test_create_request_full() -> None:
    deployment = PreviewManagedDeployment(environment="aped-1", cloud="aws", region="us-east-1")
    rc = PreviewReadCapacityOnDemandResponse(
        status=PreviewReadCapacityStatus(state="Ready")
    )
    req = PreviewCreateIndexRequest(
        schema=_make_schema(),
        name="my-index",
        deployment=deployment,
        read_capacity=rc,
        deletion_protection="enabled",
        tags={"env": "prod"},
    )
    result = msgspec.to_builtins(req)
    # omit_defaults=True: None values should not appear
    assert None not in result.values()
    assert result["name"] == "my-index"
    assert result["deletion_protection"] == "enabled"
    assert result["tags"] == {"env": "prod"}


def test_create_request_serialization_omits_none() -> None:
    req = PreviewCreateIndexRequest(schema=_make_schema(), name="idx")
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
    req = PreviewConfigureIndexRequest(schema=schema)
    result = msgspec.to_builtins(req)
    assert "schema" in result
    fields = result["schema"]["fields"]
    assert len(fields) == 2
    # Dense vector field has type discriminator "dense_vector"
    assert fields["vec"]["type"] == "dense_vector"
    # String field has type discriminator "string"
    assert fields["text"]["type"] == "string"
