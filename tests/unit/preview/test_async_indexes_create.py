"""Unit tests for AsyncPreviewIndexes.create()."""

from __future__ import annotations

import asyncio

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import PineconeValueError
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_indexes import AsyncPreviewIndexes
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
def indexes() -> AsyncPreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncPreviewIndexes(config=config)


@respx.mock
async def test_async_create_sends_post_with_api_version_header(
    indexes: AsyncPreviewIndexes,
) -> None:
    """POST /indexes carries the preview api-version header."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )

    await indexes.create(schema=_MINIMAL_SCHEMA)

    assert route.called
    request = route.calls.last.request
    assert request.url.path == "/indexes"
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
async def test_async_create_serializes_minimal_body(indexes: AsyncPreviewIndexes) -> None:
    """Minimal call sends only the schema field in the request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )

    await indexes.create(schema=_MINIMAL_SCHEMA)

    body = orjson.loads(route.calls.last.request.content)
    assert body == {"schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}}}


@respx.mock
async def test_async_create_full_body(indexes: AsyncPreviewIndexes) -> None:
    """All optional parameters are serialized into the request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )

    await indexes.create(
        schema=_MINIMAL_SCHEMA,
        name="my-index",
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
        read_capacity={"mode": "OnDemand"},
        deletion_protection="enabled",
        tags={"env": "test"},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["name"] == "my-index"
    assert body["deletion_protection"] == "enabled"
    assert body["tags"] == {"env": "test"}
    assert body["deployment"]["deployment_type"] == "managed"
    schema_field = body["schema"]["fields"]["e"]
    assert "filterable" not in schema_field


@respx.mock
async def test_async_create_returns_preview_index_model(indexes: AsyncPreviewIndexes) -> None:
    """Response JSON is deserialized into a PreviewIndexModel."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )

    result = await indexes.create(schema=_MINIMAL_SCHEMA)

    assert isinstance(result, PreviewIndexModel)
    assert result.name == "test-index"
    assert result.host == "test-index-xyz.svc.pinecone.io"
    assert result.status.state == "Initializing"
    assert result.status.ready is False


async def test_async_create_rejects_long_tag_key(indexes: AsyncPreviewIndexes) -> None:
    """Tag keys longer than 80 characters raise PineconeValueError mentioning '80'."""
    with pytest.raises(PineconeValueError, match="80"):
        await indexes.create(schema=_MINIMAL_SCHEMA, tags={"x" * 81: "v"})


async def test_async_create_rejects_long_tag_value(indexes: AsyncPreviewIndexes) -> None:
    """Tag values longer than 120 characters raise PineconeValueError mentioning '120'."""
    with pytest.raises(PineconeValueError, match="120"):
        await indexes.create(schema=_MINIMAL_SCHEMA, tags={"k": "v" * 121})


@respx.mock
async def test_async_create_unknown_field_type_passes_through(
    indexes: AsyncPreviewIndexes,
) -> None:
    """An unrecognised field type is passed through to the API without raising."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )
    await indexes.create(schema={"fields": {"x": {"type": "unknown_type"}}})
    body = orjson.loads(route.calls.last.request.content)
    assert body["schema"]["fields"]["x"] == {"type": "unknown_type"}


def test_async_create_is_coroutine() -> None:
    """AsyncPreviewIndexes.create is a coroutine function (guards against sync-drift)."""
    assert asyncio.iscoroutinefunction(AsyncPreviewIndexes.create)
