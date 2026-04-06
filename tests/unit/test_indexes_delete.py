"""Unit tests for Indexes.delete() — delete an index by name."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import Indexes
from pinecone.errors.exceptions import NotFoundError, PineconeError, ValidationError
from tests.factories import make_error_response, make_index_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture()
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture()
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


@pytest.fixture()
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent actual sleeps in polling loops."""
    monkeypatch.setattr("time.sleep", lambda _: None)


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_index_no_polling(indexes: Indexes) -> None:
    """DELETE /indexes/test-index -> 202, returns immediately (no polling)."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")

    result = indexes.delete("test-index")

    assert result is None
    # timeout=None means no polling — describe should NOT be called
    assert describe_route.call_count == 0


@respx.mock
def test_delete_removes_host_cache(indexes: Indexes) -> None:
    """Deleting an index removes its cached host URL."""
    indexes._host_cache["my-index"] = "my-index-abc.svc.pinecone.io"

    respx.delete(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(202),
    )

    indexes.delete("my-index")

    assert "my-index" not in indexes._host_cache


@respx.mock
def test_delete_polls_until_gone(indexes: Indexes, no_sleep: None) -> None:
    """With explicit timeout, poll describe until index disappears."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(200, json=make_index_response()),
            httpx.Response(404, json=make_error_response(404, "Not found")),
        ],
    )

    indexes.delete("test-index", timeout=300)

    assert describe_route.call_count == 2


# ---------------------------------------------------------------------------
# Timeout / polling
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_timeout_none_no_polling(indexes: Indexes) -> None:
    """With timeout=None (default), return immediately — describe is NOT called."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")

    indexes.delete("test-index", timeout=None)

    assert describe_route.call_count == 0


@respx.mock
def test_delete_timeout_exceeded(indexes: Indexes, no_sleep: None) -> None:
    """If index still exists after timeout, raise PineconeError."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    with pytest.raises(PineconeError, match=r"still exists after 1s"):
        indexes.delete("test-index", timeout=1)


@respx.mock
def test_delete_with_timeout_returns_when_gone(indexes: Indexes, no_sleep: None) -> None:
    """With explicit timeout, returns successfully once index is gone."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(404, json=make_error_response(404, "Not found")),
    )

    result = indexes.delete("test-index", timeout=60)

    assert result is None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_nonexistent_index(indexes: Indexes) -> None:
    """DELETE on non-existent index -> 404 -> NotFoundError."""
    respx.delete(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(404, json=make_error_response(404, "Index not found")),
    )

    with pytest.raises(NotFoundError):
        indexes.delete("no-such-index")


def test_delete_empty_name_raises(indexes: Indexes) -> None:
    """Empty name raises ValidationError before any HTTP call."""
    with pytest.raises(ValidationError):
        indexes.delete("")
