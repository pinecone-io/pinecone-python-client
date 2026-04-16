"""Unit tests for PreviewIndexes.create()."""

from __future__ import annotations

import httpx
import msgspec
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import PineconeValueError
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.indexes import PreviewIndexes
from pinecone.preview.models.indexes import PreviewIndexModel

BASE_URL = "https://api.test.pinecone.io"

_MINIMAL_SCHEMA: dict = {"fields": {"e": {"type": "dense_vector", "dimension": 4}}}

_PREVIEW_INDEX_RESPONSE: dict = {
    "name": "test-index",
    "host": "test-index-xyz.svc.pinecone.io",
    "status": {"ready": False, "state": "Initializing"},
    "schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "disabled",
}


@pytest.fixture
def indexes() -> PreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return PreviewIndexes(config=config)


@respx.mock
def test_create_sends_post_with_api_version_header(indexes: PreviewIndexes) -> None:
    """POST /indexes carries the preview api-version header."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )

    indexes.create(schema=_MINIMAL_SCHEMA)

    assert route.called
    request = route.calls.last.request
    assert request.url.path == "/indexes"
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
def test_create_serializes_minimal_body(indexes: PreviewIndexes) -> None:
    """Minimal call sends only the schema field in the request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )

    indexes.create(schema=_MINIMAL_SCHEMA)

    body = orjson.loads(route.calls.last.request.content)
    assert body == {"schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}}}


@respx.mock
def test_create_full_body(indexes: PreviewIndexes) -> None:
    """All optional parameters are serialized into the request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )

    indexes.create(
        schema=_MINIMAL_SCHEMA,
        name="my-index",
        deployment={
            "deployment_type": "managed",
            "environment": "us-east-1-aws",
            "cloud": "aws",
            "region": "us-east-1",
        },
        read_capacity={"mode": "OnDemand", "status": {"state": "Ready"}},
        deletion_protection="enabled",
        tags={"env": "test"},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["name"] == "my-index"
    assert body["deletion_protection"] == "enabled"
    assert body["tags"] == {"env": "test"}
    assert body["deployment"]["deployment_type"] == "managed"
    # filterable=False should NOT appear (omit_defaults drops False values)
    schema_field = body["schema"]["fields"]["e"]
    assert "filterable" not in schema_field


@respx.mock
def test_create_returns_preview_index_model(indexes: PreviewIndexes) -> None:
    """Response JSON is deserialized into a PreviewIndexModel."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )

    result = indexes.create(schema=_MINIMAL_SCHEMA)

    assert isinstance(result, PreviewIndexModel)
    assert result.name == "test-index"
    assert result.host == "test-index-xyz.svc.pinecone.io"
    assert result.status.state == "Initializing"
    assert result.status.ready is False


def test_create_rejects_long_tag_key(indexes: PreviewIndexes) -> None:
    """Tag keys longer than 80 characters raise PineconeValueError mentioning '80'."""
    with pytest.raises(PineconeValueError, match="80"):
        indexes.create(schema=_MINIMAL_SCHEMA, tags={"x" * 81: "v"})


def test_create_rejects_long_tag_value(indexes: PreviewIndexes) -> None:
    """Tag values longer than 120 characters raise PineconeValueError mentioning '120'."""
    with pytest.raises(PineconeValueError, match="120"):
        indexes.create(schema=_MINIMAL_SCHEMA, tags={"k": "v" * 121})


def test_create_invalid_schema_raises_validation_error(indexes: PreviewIndexes) -> None:
    """An unrecognised field type raises msgspec.ValidationError."""
    with pytest.raises(msgspec.ValidationError):
        indexes.create(schema={"fields": {"x": {"type": "unknown_type"}}})
