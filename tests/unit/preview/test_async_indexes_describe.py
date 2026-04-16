"""Unit tests for AsyncPreviewIndexes.describe()."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import NotFoundError, PineconeValueError
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_indexes import AsyncPreviewIndexes
from pinecone.preview.models.indexes import PreviewIndexModel

BASE_URL = "https://api.test.pinecone.io"

_PREVIEW_INDEX_RESPONSE: dict = {
    "name": "my-index",
    "host": "my-index-xyz.svc.pinecone.io",
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
    "error": {"code": "NOT_FOUND", "message": "Index 'missing' not found."},
    "status": 404,
}


@pytest.fixture
def indexes() -> AsyncPreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncPreviewIndexes(config=config)


@respx.mock
async def test_async_describe_sends_get_with_api_version_header(
    indexes: AsyncPreviewIndexes,
) -> None:
    """GET /indexes/{name} carries the preview api-version header."""
    route = respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    await indexes.describe("my-index")

    assert route.called
    request = route.calls.last.request
    assert request.url.path == "/indexes/my-index"
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
async def test_async_describe_returns_preview_index_model(
    indexes: AsyncPreviewIndexes,
) -> None:
    """describe() deserializes the response into a PreviewIndexModel."""
    respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    model = await indexes.describe("my-index")

    assert isinstance(model, PreviewIndexModel)
    assert model.name == "my-index"
    assert model.host == "my-index-xyz.svc.pinecone.io"
    assert model.status.ready is True


@respx.mock
async def test_async_describe_raises_not_found_on_404(indexes: AsyncPreviewIndexes) -> None:
    """describe() raises NotFoundError on a 404 response."""
    respx.get(f"{BASE_URL}/indexes/missing").mock(
        return_value=httpx.Response(404, json=_NOT_FOUND_RESPONSE)
    )

    with pytest.raises(NotFoundError):
        await indexes.describe("missing")


async def test_async_describe_raises_value_error_on_empty_name(
    indexes: AsyncPreviewIndexes,
) -> None:
    """describe("") raises PineconeValueError without issuing any HTTP call."""
    with pytest.raises(PineconeValueError):
        await indexes.describe("")
