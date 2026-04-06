"""Unit tests for Collections namespace — create, list, describe, delete."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.collections import Collections
from pinecone.errors.exceptions import NotFoundError, ValidationError
from pinecone.models.collections.collection_list import CollectionList
from pinecone.models.collections.collection_model import CollectionModel
from tests.factories import make_collection_response, make_error_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture()
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture()
def collections(http_client: HTTPClient) -> Collections:
    return Collections(http=http_client)


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
def test_create_collection(collections: Collections) -> None:
    route = respx.post(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(200, json=make_collection_response()),
    )

    result = collections.create(name="test-collection", source="my-index")

    assert isinstance(result, CollectionModel)
    assert result.name == "test-collection"
    assert result.status == "Ready"
    assert result.dimension == 1536

    # Verify request body
    request = route.calls[0].request
    body = httpx.Request("POST", "/", json={"name": "test-collection", "source": "my-index"})
    assert request.content == body.content


def test_create_empty_name_raises(collections: Collections) -> None:
    with pytest.raises(ValidationError) as exc_info:
        collections.create(name="", source="my-index")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


def test_create_empty_source_raises(collections: Collections) -> None:
    with pytest.raises(ValidationError) as exc_info:
        collections.create(name="my-collection", source="")

    assert "source" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_collections(collections: Collections) -> None:
    respx.get(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(
            200,
            json={"collections": [make_collection_response()]},
        ),
    )

    result = collections.list()

    assert isinstance(result, CollectionList)
    assert len(result) == 1
    assert result[0].name == "test-collection"
    assert result.names() == ["test-collection"]

    # verify iteration
    names = [col.name for col in result]
    assert names == ["test-collection"]


@respx.mock
def test_list_collections_empty(collections: Collections) -> None:
    respx.get(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(200, json={"collections": []}),
    )

    result = collections.list()

    assert isinstance(result, CollectionList)
    assert len(result) == 0
    assert result.names() == []


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_collection(collections: Collections) -> None:
    respx.get(f"{BASE_URL}/collections/test-collection").mock(
        return_value=httpx.Response(200, json=make_collection_response()),
    )

    result = collections.describe("test-collection")

    assert isinstance(result, CollectionModel)
    assert result.name == "test-collection"
    assert result.status == "Ready"
    assert result.size == 10_000_000
    assert result.dimension == 1536
    assert result.vector_count == 120_000
    assert result.environment == "us-east1-gcp"


@respx.mock
def test_describe_nonexistent_collection(collections: Collections) -> None:
    respx.get(f"{BASE_URL}/collections/no-such-collection").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Collection not found"),
        ),
    )

    with pytest.raises(NotFoundError):
        collections.describe("no-such-collection")


def test_describe_empty_name_raises(collections: Collections) -> None:
    with pytest.raises(ValidationError) as exc_info:
        collections.describe("")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_collection(collections: Collections) -> None:
    respx.delete(f"{BASE_URL}/collections/test-collection").mock(
        return_value=httpx.Response(202),
    )

    collections.delete("test-collection")

    # delete returns None (202 Accepted), no assertion needed


@respx.mock
def test_delete_nonexistent_collection(collections: Collections) -> None:
    respx.delete(f"{BASE_URL}/collections/no-such-collection").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Collection not found"),
        ),
    )

    with pytest.raises(NotFoundError):
        collections.delete("no-such-collection")


def test_delete_empty_name_raises(collections: Collections) -> None:
    with pytest.raises(ValidationError) as exc_info:
        collections.delete("")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# CollectionList.names() integration
# ---------------------------------------------------------------------------


@respx.mock
def test_collection_list_names(collections: Collections) -> None:
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

    result = collections.list()
    assert result.names() == ["col-a", "col-b", "col-c"]
    assert len(result) == 3
    assert result[1].name == "col-b"
