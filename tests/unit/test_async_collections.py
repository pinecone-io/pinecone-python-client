"""Unit tests for AsyncCollections namespace — create, list, describe, delete."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.collections import AsyncCollections
from pinecone.errors.exceptions import NotFoundError, ValidationError
from pinecone.models.collections.list import CollectionList
from pinecone.models.collections.model import CollectionModel
from tests.factories import make_collection_response, make_error_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def async_collections(async_http_client: AsyncHTTPClient) -> AsyncCollections:
    return AsyncCollections(http=async_http_client)


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_collection(async_collections: AsyncCollections) -> None:
    route = respx.post(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(200, json=make_collection_response()),
    )

    result = await async_collections.create(name="test-collection", source="my-index")

    assert isinstance(result, CollectionModel)
    assert result.name == "test-collection"
    assert result.status == "Ready"
    assert result.dimension == 1536

    # Verify request body
    request = route.calls[0].request
    body = httpx.Request("POST", "/", json={"name": "test-collection", "source": "my-index"})
    assert request.content == body.content


async def test_create_empty_name_raises(async_collections: AsyncCollections) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_collections.create(name="", source="my-index")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


async def test_create_empty_source_raises(async_collections: AsyncCollections) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_collections.create(name="my-collection", source="")

    assert "source" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_collections(async_collections: AsyncCollections) -> None:
    respx.get(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(
            200,
            json={"collections": [make_collection_response()]},
        ),
    )

    result = await async_collections.list()

    assert isinstance(result, CollectionList)
    assert len(result) == 1
    assert result[0].name == "test-collection"
    assert result.names() == ["test-collection"]

    # verify iteration
    names = [col.name for col in result]
    assert names == ["test-collection"]


@respx.mock
async def test_list_collections_empty(async_collections: AsyncCollections) -> None:
    respx.get(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(200, json={"collections": []}),
    )

    result = await async_collections.list()

    assert isinstance(result, CollectionList)
    assert len(result) == 0
    assert result.names() == []


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
async def test_describe_collection(async_collections: AsyncCollections) -> None:
    respx.get(f"{BASE_URL}/collections/test-collection").mock(
        return_value=httpx.Response(200, json=make_collection_response()),
    )

    result = await async_collections.describe("test-collection")

    assert isinstance(result, CollectionModel)
    assert result.name == "test-collection"
    assert result.status == "Ready"
    assert result.size == 10_000_000
    assert result.dimension == 1536
    assert result.vector_count == 120_000
    assert result.environment == "us-east1-gcp"


@respx.mock
async def test_describe_nonexistent_collection(async_collections: AsyncCollections) -> None:
    respx.get(f"{BASE_URL}/collections/no-such-collection").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Collection not found"),
        ),
    )

    with pytest.raises(NotFoundError):
        await async_collections.describe("no-such-collection")


async def test_describe_empty_name_raises(async_collections: AsyncCollections) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_collections.describe("")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
async def test_delete_collection(async_collections: AsyncCollections) -> None:
    respx.delete(f"{BASE_URL}/collections/test-collection").mock(
        return_value=httpx.Response(202),
    )

    await async_collections.delete("test-collection")

    # delete returns None (202 Accepted), no assertion needed


@respx.mock
async def test_delete_nonexistent_collection(async_collections: AsyncCollections) -> None:
    respx.delete(f"{BASE_URL}/collections/no-such-collection").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Collection not found"),
        ),
    )

    with pytest.raises(NotFoundError):
        await async_collections.delete("no-such-collection")


async def test_delete_empty_name_raises(async_collections: AsyncCollections) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_collections.delete("")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# CollectionList.names() integration
# ---------------------------------------------------------------------------


@respx.mock
async def test_collection_list_names(async_collections: AsyncCollections) -> None:
    respx.get(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(
            200,
            json={
                "collections": [
                    make_collection_response(name="col-a"),
                    make_collection_response(name="col-b"),
                    make_collection_response(name="col-c"),
                ]
            },
        ),
    )

    result = await async_collections.list()
    assert result.names() == ["col-a", "col-b", "col-c"]
    assert len(result) == 3
    assert result[1].name == "col-b"


# ---------------------------------------------------------------------------
# repr()
# ---------------------------------------------------------------------------


def test_repr(async_collections: AsyncCollections) -> None:
    assert repr(async_collections) == "AsyncCollections()"


# ---------------------------------------------------------------------------
# AsyncPinecone.collections property
# ---------------------------------------------------------------------------


def test_async_pinecone_collections_property() -> None:
    from pinecone.async_client.pinecone import AsyncPinecone

    pc = AsyncPinecone(api_key="test-key")
    collections = pc.collections
    assert isinstance(collections, AsyncCollections)
    # Verify lazy caching — same instance returned
    assert pc.collections is collections
