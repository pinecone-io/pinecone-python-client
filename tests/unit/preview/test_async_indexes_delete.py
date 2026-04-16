"""Unit tests for AsyncPreviewIndexes.delete()."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import ForbiddenError, PineconeTimeoutError, PineconeValueError
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_indexes import AsyncPreviewIndexes

BASE_URL = "https://api.test.pinecone.io"

_PREVIEW_INDEX_RESPONSE: dict = {
    "name": "x",
    "host": "x-xyz.svc.pinecone.io",
    "status": {"ready": True, "state": "Ready"},
    "schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "disabled",
}

_NOT_FOUND_RESPONSE: dict = {
    "error": {"code": "NOT_FOUND", "message": "Index 'x' not found."},
    "status": 404,
}

_FORBIDDEN_RESPONSE: dict = {
    "error": {"code": "FORBIDDEN", "message": "Deletion protection is enabled."},
    "status": 403,
}


@pytest.fixture
def indexes() -> AsyncPreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncPreviewIndexes(config=config)


@respx.mock
async def test_async_delete_returns_immediately_when_timeout_is_negative_one(
    indexes: AsyncPreviewIndexes,
) -> None:
    """delete("x", timeout=-1) returns without polling after the DELETE."""
    respx.delete(f"{BASE_URL}/indexes/x").mock(return_value=httpx.Response(204))

    with patch(
        "pinecone.preview.async_indexes.asyncio.sleep",
        side_effect=AssertionError("asyncio.sleep must not be called"),
    ), patch(
        "time.monotonic",
        side_effect=AssertionError("monotonic must not be called"),
    ):
        await indexes.delete("x", timeout=-1)


@respx.mock
async def test_async_delete_polls_until_not_found(indexes: AsyncPreviewIndexes) -> None:
    """delete() polls describe() until NotFoundError, then returns."""
    respx.delete(f"{BASE_URL}/indexes/x").mock(return_value=httpx.Response(204))

    describe_route = respx.get(f"{BASE_URL}/indexes/x")
    describe_route.side_effect = [
        httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE),
        httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE),
        httpx.Response(404, json=_NOT_FOUND_RESPONSE),
    ]

    with patch("pinecone.preview.async_indexes.asyncio.sleep", new_callable=AsyncMock):
        await indexes.delete("x")

    assert describe_route.call_count == 3


@respx.mock
async def test_async_delete_raises_timeout_error_after_deadline(
    indexes: AsyncPreviewIndexes,
) -> None:
    """delete() raises PineconeTimeoutError when timeout expires before index is gone."""
    respx.delete(f"{BASE_URL}/indexes/x").mock(return_value=httpx.Response(204))
    respx.get(f"{BASE_URL}/indexes/x").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    monotonic_values = iter([0.0, 10.0])
    with patch("pinecone.preview.async_indexes.asyncio.sleep", new_callable=AsyncMock):
        with patch("time.monotonic", side_effect=monotonic_values):
            with pytest.raises(PineconeTimeoutError, match="5s"):
                await indexes.delete("x", timeout=5)


@respx.mock
async def test_async_delete_raises_forbidden_when_protection_enabled(
    indexes: AsyncPreviewIndexes,
) -> None:
    """delete() raises ForbiddenError when the server returns 403."""
    respx.delete(f"{BASE_URL}/indexes/x").mock(
        return_value=httpx.Response(403, json=_FORBIDDEN_RESPONSE)
    )

    with pytest.raises(ForbiddenError):
        await indexes.delete("x", timeout=-1)


async def test_async_delete_raises_value_error_on_empty_name(
    indexes: AsyncPreviewIndexes,
) -> None:
    """delete("") raises PineconeValueError without issuing an HTTP request."""
    with pytest.raises(PineconeValueError):
        await indexes.delete("")


@respx.mock
async def test_async_delete_sends_api_version_header(indexes: AsyncPreviewIndexes) -> None:
    """DELETE /indexes/{name} carries the preview api-version header."""
    route = respx.delete(f"{BASE_URL}/indexes/x").mock(return_value=httpx.Response(204))

    await indexes.delete("x", timeout=-1)

    assert route.called
    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
async def test_async_delete_uses_asyncio_sleep_not_time_sleep(
    indexes: AsyncPreviewIndexes,
) -> None:
    """delete() uses asyncio.sleep (non-blocking) and never calls time.sleep."""
    respx.delete(f"{BASE_URL}/indexes/x").mock(return_value=httpx.Response(204))

    describe_route = respx.get(f"{BASE_URL}/indexes/x")
    describe_route.side_effect = [
        httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE),
        httpx.Response(404, json=_NOT_FOUND_RESPONSE),
    ]

    async_sleep_mock = AsyncMock()
    with patch("pinecone.preview.async_indexes.asyncio.sleep", async_sleep_mock), patch(
        "time.sleep", side_effect=AssertionError("time.sleep must not be called")
    ):
        await indexes.delete("x")

    async_sleep_mock.assert_awaited()


def test_async_delete_is_coroutine() -> None:
    """AsyncPreviewIndexes.delete is a coroutine function."""
    assert asyncio.iscoroutinefunction(AsyncPreviewIndexes.delete)
