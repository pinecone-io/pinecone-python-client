"""Unit tests for AsyncPreviewIndexes.configure()."""

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

_MINIMAL_SCHEMA: dict = {
    "fields": {"summary": {"type": "semantic_text", "model": "multilingual-e5-large"}}
}

_PREVIEW_INDEX_RESPONSE: dict = {
    "name": "my-index",
    "host": "my-index-xyz.svc.pinecone.io",
    "status": {"ready": True, "state": "Ready"},
    "schema": {"fields": {"summary": {"type": "semantic_text", "model": "multilingual-e5-large"}}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "enabled",
}


@pytest.fixture
def indexes() -> AsyncPreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncPreviewIndexes(config=config)


@respx.mock
async def test_async_configure_sends_patch_with_api_version_header(
    indexes: AsyncPreviewIndexes,
) -> None:
    """PATCH /indexes/{name} carries the preview api-version header."""
    route = respx.patch(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    await indexes.configure("my-index", tags={"k": "v"})

    assert route.called
    request = route.calls.last.request
    assert request.url.path == "/indexes/my-index"
    assert request.method == "PATCH"
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
async def test_async_configure_serializes_tags_only(indexes: AsyncPreviewIndexes) -> None:
    """Only the tags key appears in the request body when only tags is provided."""
    route = respx.patch(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    await indexes.configure("my-index", tags={"env": "prod"})

    body = orjson.loads(route.calls.last.request.content)
    assert body == {"tags": {"env": "prod"}}


@respx.mock
async def test_async_configure_serializes_schema_only(indexes: AsyncPreviewIndexes) -> None:
    """Only the schema key appears in the request body when only schema is provided."""
    route = respx.patch(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    await indexes.configure("my-index", schema=_MINIMAL_SCHEMA)

    body = orjson.loads(route.calls.last.request.content)
    assert list(body.keys()) == ["schema"]
    assert body["schema"]["fields"]["summary"]["type"] == "semantic_text"
    assert body["schema"]["fields"]["summary"]["model"] == "multilingual-e5-large"


@respx.mock
async def test_async_configure_serializes_all_fields(indexes: AsyncPreviewIndexes) -> None:
    """Body contains all four top-level keys when all parameters are provided."""
    route = respx.patch(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    await indexes.configure(
        "my-index",
        schema=_MINIMAL_SCHEMA,
        deletion_protection="enabled",
        tags={"env": "prod"},
        read_capacity={"mode": "OnDemand"},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert set(body.keys()) == {"schema", "deletion_protection", "tags", "read_capacity"}


@respx.mock
async def test_async_configure_returns_preview_index_model(
    indexes: AsyncPreviewIndexes,
) -> None:
    """200 response is deserialized into a PreviewIndexModel with correct fields."""
    respx.patch(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    result = await indexes.configure("my-index", deletion_protection="enabled")

    assert isinstance(result, PreviewIndexModel)
    assert result.name == "my-index"
    assert result.deletion_protection == "enabled"
    assert result.host == "my-index-xyz.svc.pinecone.io"


async def test_async_configure_rejects_empty_name(indexes: AsyncPreviewIndexes) -> None:
    """An empty name raises PineconeValueError without making an HTTP call."""
    with pytest.raises(PineconeValueError):
        await indexes.configure("", tags={"a": "b"})


async def test_async_configure_rejects_all_none(indexes: AsyncPreviewIndexes) -> None:
    """configure("x") with no kwargs raises PineconeValueError mentioning 'at least one'."""
    with pytest.raises(PineconeValueError, match="at least one"):
        await indexes.configure("x")


async def test_async_configure_rejects_empty_schema_dict(indexes: AsyncPreviewIndexes) -> None:
    """An empty schema dict raises PineconeValueError mentioning 'schema'."""
    with pytest.raises(PineconeValueError, match="schema"):
        await indexes.configure("x", schema={})


async def test_async_configure_rejects_empty_tags_dict(indexes: AsyncPreviewIndexes) -> None:
    """An empty tags dict raises PineconeValueError."""
    with pytest.raises(PineconeValueError):
        await indexes.configure("x", tags={})


async def test_async_configure_rejects_empty_read_capacity_dict(
    indexes: AsyncPreviewIndexes,
) -> None:
    """An empty read_capacity dict raises PineconeValueError."""
    with pytest.raises(PineconeValueError):
        await indexes.configure("x", read_capacity={})


async def test_async_configure_rejects_long_tag_key(indexes: AsyncPreviewIndexes) -> None:
    """A tag key exceeding 80 characters raises PineconeValueError mentioning '80'."""
    with pytest.raises(PineconeValueError, match="80"):
        await indexes.configure("my-index", tags={"x" * 81: "v"})


async def test_async_configure_rejects_long_tag_value(indexes: AsyncPreviewIndexes) -> None:
    """A tag value exceeding 120 characters raises PineconeValueError mentioning '120'."""
    with pytest.raises(PineconeValueError, match="120"):
        await indexes.configure("my-index", tags={"x": "v" * 121})


def test_async_configure_is_coroutine() -> None:
    """AsyncPreviewIndexes.configure is a coroutine function."""
    assert asyncio.iscoroutinefunction(AsyncPreviewIndexes.configure)


async def test_async_configure_deployment_empty_dict_raises(
    indexes: AsyncPreviewIndexes,
) -> None:
    """An empty deployment dict raises PineconeValueError before any HTTP call."""
    with pytest.raises(PineconeValueError, match="deployment"):
        await indexes.configure("my-index", deployment={})


@respx.mock
async def test_async_configure_deployment_sends_correct_body(
    indexes: AsyncPreviewIndexes,
) -> None:
    """configure(deployment={"replicas": 2}) serializes deployment into the request body."""
    route = respx.patch(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    await indexes.configure("my-index", deployment={"replicas": 2})

    body = orjson.loads(route.calls.last.request.content)
    assert body == {"deployment": {"replicas": 2}}


async def test_async_configure_schema_non_semantic_text_raises(
    indexes: AsyncPreviewIndexes,
) -> None:
    """configure() with a non-semantic_text schema field raises PineconeValueError before HTTP."""
    with pytest.raises(PineconeValueError, match="dense_vector"):
        await indexes.configure("idx", schema={"fields": {"vec": {"type": "dense_vector"}}})


@respx.mock
async def test_async_configure_schema_semantic_text_accepted(
    indexes: AsyncPreviewIndexes,
) -> None:
    """configure() with a semantic_text schema field succeeds and does not raise PineconeValueError."""
    respx.patch(f"{BASE_URL}/indexes/idx").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    result = await indexes.configure(
        "idx",
        schema={"fields": {"summary": {"type": "semantic_text", "model": "multilingual-e5-large"}}},
    )
    assert isinstance(result, PreviewIndexModel)
