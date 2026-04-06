"""Unit tests for index creation/deletion polling edge cases."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import _POLL_INTERVAL_SECONDS, Indexes
from pinecone.errors.exceptions import (
    ConflictError,
    IndexInitFailedError,
    NotFoundError,
    PineconeTimeoutError,
)
from pinecone.models.indexes.index import IndexModel
from tests.factories import make_error_response, make_index_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture()
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture()
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


# ---------------------------------------------------------------------------
# Polling interval constant
# ---------------------------------------------------------------------------


def test_poll_interval_is_five_seconds() -> None:
    """The polling interval is a fixed 5-second constant."""
    assert _POLL_INTERVAL_SECONDS == 5


# ---------------------------------------------------------------------------
# Create: happy path polling
# ---------------------------------------------------------------------------


@respx.mock
def test_create_polls_until_ready(indexes: Indexes) -> None:
    """Index transitions to ready -> returns IndexModel."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(
                200,
                json=make_index_response(status={"ready": False, "state": "Initializing"}),
            ),
            httpx.Response(
                200,
                json=make_index_response(status={"ready": True, "state": "Ready"}),
            ),
        ]
    )

    with patch("pinecone.client.indexes.time.sleep") as mock_sleep:
        result = indexes.create(
            name="test-index",
            dimension=1536,
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=300,
        )

    assert isinstance(result, IndexModel)
    assert result.status.ready is True
    mock_sleep.assert_called_with(_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Create: InitializationFailed
# ---------------------------------------------------------------------------


@respx.mock
def test_create_init_failed_raises_immediately(indexes: Indexes) -> None:
    """Index transitions to InitializationFailed -> raises IndexInitFailedError."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(status={"ready": False, "state": "InitializationFailed"}),
        ),
    )

    with patch("pinecone.client.indexes.time.sleep"):
        with pytest.raises(IndexInitFailedError) as exc_info:
            indexes.create(
                name="test-index",
                dimension=1536,
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
                timeout=300,
            )

    assert exc_info.value.index_name == "test-index"
    assert "InitializationFailed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Create: timeout exceeded
# ---------------------------------------------------------------------------


@respx.mock
def test_create_timeout_raises(indexes: Indexes) -> None:
    """Polling exceeds timeout -> raises PineconeTimeoutError."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )

    with patch("pinecone.client.indexes.time.sleep"):
        with pytest.raises(PineconeTimeoutError, match="not ready after"):
            indexes.create(
                name="test-index",
                dimension=1536,
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
                timeout=1,
            )


# ---------------------------------------------------------------------------
# Create: timeout=-1 skips polling
# ---------------------------------------------------------------------------


@respx.mock
def test_create_timeout_negative_one_skips_polling(indexes: Indexes) -> None:
    """timeout=-1 returns immediately without polling."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")

    result = indexes.create(
        name="test-index",
        dimension=1536,
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        timeout=-1,
    )

    assert isinstance(result, IndexModel)
    assert result.status.ready is False
    assert describe_route.call_count == 0


# ---------------------------------------------------------------------------
# Polling uses exactly 5-second sleep
# ---------------------------------------------------------------------------


@respx.mock
def test_polling_sleep_interval(indexes: Indexes) -> None:
    """Verify time.sleep is called with exactly _POLL_INTERVAL_SECONDS."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(
                200,
                json=make_index_response(status={"ready": False, "state": "Initializing"}),
            ),
            httpx.Response(
                200,
                json=make_index_response(status={"ready": True, "state": "Ready"}),
            ),
        ]
    )

    with patch("pinecone.client.indexes.time.sleep") as mock_sleep:
        indexes.create(
            name="test-index",
            dimension=1536,
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=300,
        )

    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_with(5)


# ---------------------------------------------------------------------------
# Describe non-existent index -> NotFoundError
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_nonexistent_raises_not_found(indexes: Indexes) -> None:
    """Describing a non-existent index propagates NotFoundError from HTTP layer."""
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Index not found"),
        ),
    )

    with pytest.raises(NotFoundError):
        indexes.describe("no-such-index")


# ---------------------------------------------------------------------------
# Create duplicate index -> ConflictError
# ---------------------------------------------------------------------------


@respx.mock
def test_create_duplicate_raises_conflict(indexes: Indexes) -> None:
    """Creating an index with an existing name propagates ConflictError from HTTP layer."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            409,
            json=make_error_response(409, "Index already exists"),
        ),
    )

    with pytest.raises(ConflictError):
        indexes.create(
            name="existing-index",
            dimension=1536,
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )


# ---------------------------------------------------------------------------
# Delete: timeout exceeded
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_timeout_raises(indexes: Indexes) -> None:
    """Delete polling exceeds timeout -> raises PineconeTimeoutError."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    with patch("pinecone.client.indexes.time.sleep"):
        with pytest.raises(PineconeTimeoutError, match="still exists after"):
            indexes.delete("test-index", timeout=1)


# ---------------------------------------------------------------------------
# Delete: polls until gone
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_polls_until_not_found(indexes: Indexes) -> None:
    """Delete polling succeeds when index returns 404."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(200, json=make_index_response()),
            httpx.Response(404, json=make_error_response(404, "Not found")),
        ],
    )

    with patch("pinecone.client.indexes.time.sleep"):
        indexes.delete("test-index", timeout=60)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_index_init_failed_is_pinecone_error() -> None:
    """IndexInitFailedError is a subclass of PineconeError."""
    from pinecone.errors.exceptions import PineconeError

    err = IndexInitFailedError("my-index")
    assert isinstance(err, PineconeError)
    assert err.index_name == "my-index"


def test_pinecone_timeout_is_pinecone_error() -> None:
    """PineconeTimeoutError is a subclass of PineconeError."""
    from pinecone.errors.exceptions import PineconeError

    err = PineconeTimeoutError("timed out")
    assert isinstance(err, PineconeError)
