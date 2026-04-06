"""Unit tests for AsyncIndexes namespace — list, describe, exists."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.errors.exceptions import NotFoundError, ValidationError
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList
from tests.factories import (
    make_error_response,
    make_index_list_response,
    make_index_response,
)

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture()
def async_http_client() -> AsyncHTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture()
def async_indexes(async_http_client: AsyncHTTPClient) -> AsyncIndexes:
    return AsyncIndexes(http=async_http_client)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_indexes(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response()),
    )

    result = await async_indexes.list()

    assert isinstance(result, IndexList)
    assert len(result) == 1
    assert result[0].name == "test-index"
    assert result.names() == ["test-index"]

    # verify iteration
    names = [idx.name for idx in result]
    assert names == ["test-index"]


@respx.mock
async def test_list_indexes_empty(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": []}),
    )

    result = await async_indexes.list()

    assert isinstance(result, IndexList)
    assert len(result) == 0
    assert result.names() == []


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
async def test_describe_index(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    result = await async_indexes.describe("test-index")

    assert isinstance(result, IndexModel)
    assert result.name == "test-index"
    assert result.dimension == 1536
    assert result.metric == "cosine"
    assert result.host == "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
    assert result.vector_type == "dense"
    assert result.deletion_protection == "disabled"
    assert result.status.ready is True
    assert result.status.state == "Ready"
    # bracket access
    assert result["name"] == "test-index"
    assert result["dimension"] == 1536


@respx.mock
async def test_describe_index_caches_host(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    await async_indexes.describe("test-index")

    assert "test-index" in async_indexes._host_cache
    expected_host = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
    assert async_indexes._host_cache["test-index"] == expected_host


@respx.mock
async def test_describe_nonexistent_index(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Index not found"),
        ),
    )

    with pytest.raises(NotFoundError):
        await async_indexes.describe("no-such-index")


async def test_describe_empty_name_raises(async_indexes: AsyncIndexes) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.describe("")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


@respx.mock
async def test_exists_true(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    assert await async_indexes.exists("test-index") is True


@respx.mock
async def test_exists_false(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Index not found"),
        ),
    )

    assert await async_indexes.exists("no-such-index") is False


async def test_exists_empty_name_raises(async_indexes: AsyncIndexes) -> None:
    with pytest.raises(ValidationError):
        await async_indexes.exists("")
