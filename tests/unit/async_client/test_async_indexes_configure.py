"""Unit tests for AsyncIndexes.configure() — embed parameter handling."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.indexes import AsyncIndexes

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def async_http_client() -> AsyncHTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def idx(async_http_client: AsyncHTTPClient) -> AsyncIndexes:
    return AsyncIndexes(http=async_http_client)


@pytest.mark.asyncio
async def test_configure_forwards_embed_to_patch_body(idx: AsyncIndexes) -> None:
    """embed dict is set as 'embed' key in the PATCH body."""
    patch = AsyncMock()
    idx._http.patch = patch  # type: ignore[method-assign]

    await idx.configure("my-index", embed={"model": "new-model"})

    patch.assert_awaited_once()
    assert patch.await_args.kwargs["json"]["embed"] == {"model": "new-model"}


@pytest.mark.asyncio
async def test_configure_omits_embed_when_none(idx: AsyncIndexes) -> None:
    """When embed is not provided, 'embed' key is absent from the PATCH body."""
    patch = AsyncMock()
    idx._http.patch = patch  # type: ignore[method-assign]

    await idx.configure("my-index", replicas=2)

    patch.assert_awaited_once()
    assert "embed" not in patch.await_args.kwargs["json"]
