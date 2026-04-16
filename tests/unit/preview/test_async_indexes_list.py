"""Unit tests for AsyncPreviewIndexes.list()."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.pagination import AsyncPaginator
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_indexes import AsyncPreviewIndexes
from pinecone.preview.models.indexes import PreviewIndexModel

BASE_URL = "https://api.test.pinecone.io"

_INDEX_1: dict = {
    "name": "index-one",
    "host": "index-one-xyz.svc.pinecone.io",
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

_INDEX_2: dict = {
    "name": "index-two",
    "host": "index-two-xyz.svc.pinecone.io",
    "status": {"ready": False, "state": "Initializing"},
    "schema": {"fields": {"v": {"type": "dense_vector", "dimension": 8}}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "disabled",
}

_INDEX_3: dict = {
    "name": "index-three",
    "host": "index-three-xyz.svc.pinecone.io",
    "status": {"ready": True, "state": "Ready"},
    "schema": {"fields": {"w": {"type": "dense_vector", "dimension": 16}}},
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
async def test_async_list_returns_async_paginator(indexes: AsyncPreviewIndexes) -> None:
    """list() returns an AsyncPaginator instance without awaiting."""
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [_INDEX_1]})
    )

    result = indexes.list()

    assert isinstance(result, AsyncPaginator)


@respx.mock
async def test_async_list_iterates_items(indexes: AsyncPreviewIndexes) -> None:
    """async for over await list() yields PreviewIndexModel instances."""
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [_INDEX_1, _INDEX_2, _INDEX_3]})
    )

    items: list[PreviewIndexModel] = []
    async for idx in indexes.list():
        items.append(idx)

    assert len(items) == 3
    assert all(isinstance(idx, PreviewIndexModel) for idx in items)
    assert items[0].name == "index-one"
    assert items[1].name == "index-two"
    assert items[2].name == "index-three"


@respx.mock
async def test_async_list_paginator_to_list(indexes: AsyncPreviewIndexes) -> None:
    """await (await list()).to_list() collects all items."""
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [_INDEX_1, _INDEX_2]})
    )

    items = await indexes.list().to_list()

    assert len(items) == 2
    assert all(isinstance(item, PreviewIndexModel) for item in items)


@respx.mock
async def test_async_list_sends_api_version_header(indexes: AsyncPreviewIndexes) -> None:
    """list() sends X-Pinecone-Api-Version: 2026-01.alpha."""
    route = respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [_INDEX_1]})
    )

    async for _ in indexes.list():
        break

    assert route.called
    request = route.calls.last.request
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
async def test_async_list_rejects_non_positive_limit(
    respx_mock: respx.MockRouter, indexes: AsyncPreviewIndexes
) -> None:
    """list(limit=0) raises PineconeValueError before any HTTP call."""
    with pytest.raises(PineconeValueError):
        indexes.list(limit=0)

    assert respx_mock.calls.call_count == 0


@respx.mock
async def test_async_list_terminates_after_one_page(indexes: AsyncPreviewIndexes) -> None:
    """Iterating fully issues exactly one GET /indexes request."""
    route = respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [_INDEX_1, _INDEX_2]})
    )

    async for _ in indexes.list():
        pass

    assert route.call_count == 1


def test_async_list_is_not_coroutine() -> None:
    """AsyncPreviewIndexes.list is a plain function, not a coroutine."""
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    indexes = AsyncPreviewIndexes(config=config)
    assert not asyncio.iscoroutinefunction(AsyncPreviewIndexes.list)
    assert isinstance(indexes.list(), AsyncPaginator)
