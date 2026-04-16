"""Unit tests for PreviewIndexes.configure()."""

from __future__ import annotations

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import PineconeValueError
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.indexes import PreviewIndexes
from pinecone.preview.models.indexes import PreviewIndexModel

BASE_URL = "https://api.test.pinecone.io"

_MINIMAL_SCHEMA: dict = {"fields": {"emb": {"type": "dense_vector", "dimension": 4}}}

_PREVIEW_INDEX_RESPONSE: dict = {
    "name": "my",
    "host": "my-xyz.svc.pinecone.io",
    "status": {"ready": True, "state": "Ready"},
    "schema": {"fields": {"emb": {"type": "dense_vector", "dimension": 4}}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "enabled",
}


@pytest.fixture
def indexes() -> PreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return PreviewIndexes(config=config)


@respx.mock
def test_configure_sends_patch_with_api_version(indexes: PreviewIndexes) -> None:
    """PATCH /indexes/{name} carries the preview api-version header."""
    route = respx.patch(f"{BASE_URL}/indexes/my").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    indexes.configure("my", deletion_protection="enabled")

    assert route.called
    request = route.calls.last.request
    assert request.url.path == "/indexes/my"
    assert request.method == "PATCH"
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
def test_configure_serializes_only_provided_fields(indexes: PreviewIndexes) -> None:
    """Only the provided kwarg appears in the request body."""
    route = respx.patch(f"{BASE_URL}/indexes/my").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    indexes.configure("my", tags={"env": "prod"})

    body = orjson.loads(route.calls.last.request.content)
    assert body == {"tags": {"env": "prod"}}


@respx.mock
def test_configure_schema_only(indexes: PreviewIndexes) -> None:
    """Schema-only call sends only the schema key; valid field values survive the None filter."""
    route = respx.patch(f"{BASE_URL}/indexes/my").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    indexes.configure("my", schema=_MINIMAL_SCHEMA)

    body = orjson.loads(route.calls.last.request.content)
    assert list(body.keys()) == ["schema"]
    assert body["schema"]["fields"]["emb"]["type"] == "dense_vector"
    assert body["schema"]["fields"]["emb"]["dimension"] == 4


def test_configure_requires_at_least_one_parameter(indexes: PreviewIndexes) -> None:
    """Calling configure with no kwargs raises PineconeValueError mentioning 'at least one'."""
    with pytest.raises(PineconeValueError, match="at least one"):
        indexes.configure("x")


def test_configure_rejects_empty_schema_dict(indexes: PreviewIndexes) -> None:
    """An empty schema dict raises PineconeValueError mentioning 'schema'."""
    with pytest.raises(PineconeValueError, match="schema"):
        indexes.configure("x", schema={})


def test_configure_rejects_empty_tags_dict(indexes: PreviewIndexes) -> None:
    """An empty tags dict raises PineconeValueError."""
    with pytest.raises(PineconeValueError):
        indexes.configure("x", tags={})


def test_configure_rejects_empty_read_capacity_dict(indexes: PreviewIndexes) -> None:
    """An empty read_capacity dict raises PineconeValueError."""
    with pytest.raises(PineconeValueError):
        indexes.configure("x", read_capacity={})


def test_configure_rejects_empty_name(indexes: PreviewIndexes) -> None:
    """An empty name raises PineconeValueError without making an HTTP call."""
    with pytest.raises(PineconeValueError):
        indexes.configure("", tags={"a": "b"})


def test_configure_rejects_long_tag_key(indexes: PreviewIndexes) -> None:
    """A tag key exceeding 80 characters raises PineconeValueError mentioning '80'."""
    with pytest.raises(PineconeValueError, match="80"):
        indexes.configure("my", tags={"x" * 81: "v"})


def test_configure_rejects_long_tag_value(indexes: PreviewIndexes) -> None:
    """Tag values longer than 120 characters raise PineconeValueError mentioning '120'."""
    with pytest.raises(PineconeValueError, match="120"):
        indexes.configure("my", tags={"k": "v" * 121})


@respx.mock
def test_configure_returns_updated_index_model(indexes: PreviewIndexes) -> None:
    """Response JSON is deserialized into a PreviewIndexModel."""
    respx.patch(f"{BASE_URL}/indexes/my").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    result = indexes.configure("my", deletion_protection="enabled")

    assert isinstance(result, PreviewIndexModel)
    assert result.name == "my"
    assert result.deletion_protection == "enabled"
    assert result.host == "my-xyz.svc.pinecone.io"
