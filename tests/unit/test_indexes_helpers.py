"""Unit tests for index helper functions."""

from __future__ import annotations

from pinecone._internal.indexes_helpers import build_create_body


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
