"""Unit tests for index helper functions."""

from __future__ import annotations

import pytest

from pinecone._internal.indexes_helpers import build_create_body, validate_read_capacity
from pinecone.errors.exceptions import ValidationError


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


def test_validate_read_capacity_missing_both_shards_error_first() -> None:
    """When both shards and replicas are absent, missing-shards error is raised first."""
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "g2-standard-8",
            "scaling": "Manual",
            "manual": {},
        },
    }
    with pytest.raises(ValidationError, match="shards"):
        validate_read_capacity(read_capacity)


def test_validate_read_capacity_missing_only_replicas() -> None:
    """When only replicas is absent, the replicas error is raised."""
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "g2-standard-8",
            "scaling": "Manual",
            "manual": {"shards": 1},
        },
    }
    with pytest.raises(ValidationError, match="replicas"):
        validate_read_capacity(read_capacity)


def test_validate_read_capacity_missing_only_shards() -> None:
    """When only shards is absent, the shards error is raised."""
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "g2-standard-8",
            "scaling": "Manual",
            "manual": {"replicas": 2},
        },
    }
    with pytest.raises(ValidationError, match="shards"):
        validate_read_capacity(read_capacity)
