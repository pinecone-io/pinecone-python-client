"""Unit tests for index helper functions."""

from __future__ import annotations

import pytest

from pinecone._internal.indexes_helpers import (
    build_create_body,
    build_integrated_body,
    validate_read_capacity,
)
from pinecone.errors.exceptions import ValidationError
from pinecone.models.indexes.specs import EmbedConfig, IntegratedSpec, ServerlessSpec


def test_build_create_body_dict_spec_not_mutated_by_schema() -> None:
    """Passing a dict spec with a schema must not mutate the caller's original dict."""
    spec = {"serverless": {"cloud": "aws", "region": "us-east-1"}}

    build_create_body(
        name="test",
        spec=spec,
        dimension=3,
        metric="cosine",
        vector_type="dense",
        deletion_protection="disabled",
        tags=None,
        schema={"fields": {"genre": "string"}},
    )

    assert "schema" not in spec["serverless"]


def test_validate_read_capacity_dedicated_partial_shards_only() -> None:
    """Partial patch with only shards specified must not raise."""
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {"scaling": "Manual", "manual": {"shards": 4}},
    }
    validate_read_capacity(read_capacity)


def test_validate_read_capacity_dedicated_partial_replicas_only() -> None:
    """Partial patch with only replicas specified must not raise."""
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {"scaling": "Manual", "manual": {"replicas": 2}},
    }
    validate_read_capacity(read_capacity)


def test_validate_read_capacity_dedicated_no_node_type() -> None:
    """Omitting node_type from a Dedicated patch must not raise."""
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {},
    }
    validate_read_capacity(read_capacity)


def test_validate_read_capacity_on_demand_no_raise() -> None:
    """OnDemand mode requires no further fields."""
    validate_read_capacity({"mode": "OnDemand"})


def test_validate_read_capacity_missing_mode_raises() -> None:
    """Omitting mode entirely must raise ValidationError."""
    with pytest.raises(ValidationError, match="mode"):
        validate_read_capacity({})


def test_validate_read_capacity_dedicated_wrong_type_raises() -> None:
    """Supplying a non-dict for 'dedicated' must raise ValidationError."""
    with pytest.raises(ValidationError, match="dedicated"):
        validate_read_capacity({"mode": "Dedicated", "dedicated": "string"})


def _make_integrated_spec() -> IntegratedSpec:
    return IntegratedSpec(
        cloud="aws",
        region="us-east-1",
        embed=EmbedConfig(model="multilingual-e5-large", field_map={"text": "body"}),
    )


def test_build_integrated_body_includes_schema() -> None:
    body = build_integrated_body(
        name="my-index",
        spec=_make_integrated_spec(),
        deletion_protection="disabled",
        tags=None,
        schema={"body": {"type": "str"}},
    )
    assert body["schema"] == {"body": {"type": "str"}}


def test_build_integrated_body_schema_absent_when_none() -> None:
    body = build_integrated_body(
        name="my-index",
        spec=_make_integrated_spec(),
        deletion_protection="disabled",
        tags=None,
        schema=None,
    )
    assert "schema" not in body


def test_build_integrated_body_includes_read_capacity() -> None:
    rc = {"mode": "OnDemand"}
    body = build_integrated_body(
        name="my-index",
        spec=_make_integrated_spec(),
        deletion_protection="disabled",
        tags=None,
        read_capacity=rc,
    )
    assert body["read_capacity"] == rc


def test_build_integrated_body_read_capacity_absent_when_none() -> None:
    body = build_integrated_body(
        name="my-index",
        spec=_make_integrated_spec(),
        deletion_protection="disabled",
        tags=None,
        read_capacity=None,
    )
    assert "read_capacity" not in body


def test_build_create_body_serverless_includes_read_capacity() -> None:
    spec = ServerlessSpec(
        cloud="aws",
        region="us-east-1",
        read_capacity={
            "mode": "Dedicated",
            "dedicated": {
                "node_type": "t1",
                "scaling": "Manual",
                "manual": {"shards": 2, "replicas": 3},
            },
        },
    )
    body = build_create_body(
        name="test-index",
        spec=spec,
        dimension=128,
        metric="cosine",
        vector_type="dense",
        deletion_protection="disabled",
        tags=None,
        schema=None,
    )
    assert body["spec"]["serverless"]["read_capacity"]["mode"] == "Dedicated"


def test_build_create_body_serverless_read_capacity_absent_when_none() -> None:
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    body = build_create_body(
        name="test-index",
        spec=spec,
        dimension=128,
        metric="cosine",
        vector_type="dense",
        deletion_protection="disabled",
        tags=None,
        schema=None,
    )
    assert "read_capacity" not in body["spec"]["serverless"]


def test_build_create_body_serverless_spec_schema_included() -> None:
    spec = ServerlessSpec(cloud="aws", region="us-east-1", schema={"genre": {"type": "str"}})
    body = build_create_body(
        name="test-index",
        spec=spec,
        dimension=128,
        metric="cosine",
        vector_type="dense",
        deletion_protection="disabled",
        tags=None,
        schema=None,
    )
    assert body["spec"]["serverless"]["schema"] == {"genre": {"type": "str"}}
