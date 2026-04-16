"""Unit tests for AsyncPreview namespace — lazy initialization and wiring."""

from __future__ import annotations

import sys

import httpx
import pytest
import respx

from pinecone import AsyncPinecone
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_indexes import AsyncPreviewIndexes
from pinecone.preview.models.indexes import PreviewIndexModel

BASE_URL = "https://api.test.pinecone.io"

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


def test_async_preview_indexes_property_returns_async_preview_indexes_instance() -> None:
    pc = AsyncPinecone(api_key="test")
    assert isinstance(pc.preview.indexes, AsyncPreviewIndexes)


def test_async_preview_indexes_property_is_cached() -> None:
    pc = AsyncPinecone(api_key="test")
    first = pc.preview.indexes
    second = pc.preview.indexes
    assert first is second


def test_async_preview_property_is_cached() -> None:
    pc = AsyncPinecone(api_key="test")
    first = pc.preview
    second = pc.preview
    assert first is second


def test_async_preview_indexes_lazy_until_accessed() -> None:
    pc = AsyncPinecone(api_key="test")
    preview = pc.preview
    assert preview._indexes is None
    _ = preview.indexes
    assert preview._indexes is not None


@respx.mock
@pytest.mark.asyncio
async def test_async_preview_indexes_end_to_end_create() -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )
    pc = AsyncPinecone(api_key="test-key", host=BASE_URL)
    result = await pc.preview.indexes.create(
        schema={"fields": {"e": {"type": "dense_vector", "dimension": 4}}}
    )
    assert route.called
    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION
    assert isinstance(result, PreviewIndexModel)


def test_async_pinecone_init_does_not_import_async_preview_indexes() -> None:
    saved = sys.modules.pop("pinecone.preview.async_indexes", None)
    try:
        pc = AsyncPinecone(api_key="test")
        _ = pc.preview
        assert "pinecone.preview.async_indexes" not in sys.modules
        _ = pc.preview.indexes
        assert "pinecone.preview.async_indexes" in sys.modules
    finally:
        if saved is not None:
            sys.modules["pinecone.preview.async_indexes"] = saved
