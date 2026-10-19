"""Unit tests for the ``timeout`` argument of create / create_for_model / delete.

A negative ``timeout`` other than ``-1`` is rejected before any request is
sent, and each poll sleep is clamped to the time left before the deadline.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.client.indexes import Indexes
from pinecone.errors.exceptions import PineconeTimeoutError, PineconeValueError
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"
SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 4, "metric": "cosine"}}
}
EMBED: dict[str, Any] = {"model": "multilingual-e5-large", "field_map": {"text": "chunk_text"}}
NOT_READY = {"ready": False, "state": "Initializing"}

BAD_TIMEOUTS = [-2, -5, -0.5, -3600]


class FakeClock:
    """Monotonic clock that only advances when a sleep is requested."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    async def async_sleep(self, seconds: float) -> None:
        self.sleep(seconds)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("time.monotonic", fake.monotonic)
    monkeypatch.setattr("time.sleep", fake.sleep)
    monkeypatch.setattr("asyncio.sleep", fake.async_sleep)
    return fake


@pytest.fixture
def indexes() -> Indexes:
    return Indexes(
        http=HTTPClient(PineconeConfig(api_key="k", host=BASE_URL), CONTROL_PLANE_API_VERSION)
    )


@pytest.fixture
async def async_indexes() -> AsyncGenerator[AsyncIndexes]:
    http = AsyncHTTPClient(PineconeConfig(api_key="k", host=BASE_URL), CONTROL_PLANE_API_VERSION)
    yield AsyncIndexes(http=http)
    await http.close()


def _mock_never_ready() -> dict[str, respx.Route]:
    return {
        "create": respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(201, json=make_index_response(status=NOT_READY))
        ),
        "create_for_model": respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
            return_value=httpx.Response(201, json=make_index_response(status=NOT_READY))
        ),
        "delete": respx.delete(f"{BASE_URL}/indexes/test-index").mock(
            return_value=httpx.Response(202)
        ),
        "describe": respx.get(f"{BASE_URL}/indexes/test-index").mock(
            return_value=httpx.Response(200, json=make_index_response(status=NOT_READY))
        ),
    }


SYNC_CALLS: dict[str, Callable[[Indexes, Any], Any]] = {
    "create": lambda ix, t: ix.create(name="test-index", schema=SCHEMA, timeout=t),
    "create_for_model": lambda ix, t: ix.create_for_model(
        name="test-index", cloud="aws", region="us-east-1", embed=EMBED, timeout=t
    ),
    "delete": lambda ix, t: ix.delete("test-index", timeout=t),
}

ASYNC_CALLS: dict[str, Callable[[AsyncIndexes, Any], Any]] = {
    "create": lambda ix, t: ix.create(name="test-index", schema=SCHEMA, timeout=t),
    "create_for_model": lambda ix, t: ix.create_for_model(
        name="test-index", cloud="aws", region="us-east-1", embed=EMBED, timeout=t
    ),
    "delete": lambda ix, t: ix.delete("test-index", timeout=t),
}


@pytest.mark.parametrize("timeout", BAD_TIMEOUTS)
@pytest.mark.parametrize("method", list(SYNC_CALLS))
@respx.mock
def test_sync_negative_timeout_raises_before_any_request(
    indexes: Indexes, method: str, timeout: float
) -> None:
    routes = _mock_never_ready()

    with pytest.raises(PineconeValueError, match="timeout"):
        SYNC_CALLS[method](indexes, timeout)

    assert respx.calls.call_count == 0
    assert all(not r.called for r in routes.values())


@pytest.mark.parametrize("timeout", BAD_TIMEOUTS)
@pytest.mark.parametrize("method", list(ASYNC_CALLS))
@respx.mock
async def test_async_negative_timeout_raises_before_any_request(
    async_indexes: AsyncIndexes, method: str, timeout: float
) -> None:
    routes = _mock_never_ready()

    with pytest.raises(PineconeValueError, match="timeout"):
        await ASYNC_CALLS[method](async_indexes, timeout)

    assert respx.calls.call_count == 0
    assert all(not r.called for r in routes.values())


@pytest.mark.parametrize("method", ["create", "create_for_model", "delete"])
@respx.mock
def test_sync_minus_one_still_skips_polling(indexes: Indexes, method: str) -> None:
    routes = _mock_never_ready()

    SYNC_CALLS[method](indexes, -1)

    assert not routes["describe"].called


@pytest.mark.parametrize("method", ["create", "create_for_model", "delete"])
@respx.mock
async def test_async_minus_one_still_skips_polling(
    async_indexes: AsyncIndexes, method: str
) -> None:
    routes = _mock_never_ready()

    await ASYNC_CALLS[method](async_indexes, -1)

    assert not routes["describe"].called


@pytest.mark.parametrize("method", list(SYNC_CALLS))
@respx.mock
def test_sync_poll_sleep_is_clamped_to_deadline(
    indexes: Indexes, clock: FakeClock, method: str
) -> None:
    _mock_never_ready()

    with pytest.raises(PineconeTimeoutError):
        SYNC_CALLS[method](indexes, 1)

    assert clock.sleeps == [1.0]
    assert clock.now == 1.0


@pytest.mark.parametrize("method", list(ASYNC_CALLS))
@respx.mock
async def test_async_poll_sleep_is_clamped_to_deadline(
    async_indexes: AsyncIndexes, clock: FakeClock, method: str
) -> None:
    _mock_never_ready()

    with pytest.raises(PineconeTimeoutError):
        await ASYNC_CALLS[method](async_indexes, 1)

    assert clock.sleeps == [1.0]
    assert clock.now == 1.0


@respx.mock
def test_sync_poll_sleeps_full_interval_until_last_partial_step(
    indexes: Indexes, clock: FakeClock
) -> None:
    _mock_never_ready()

    with pytest.raises(PineconeTimeoutError):
        indexes.create(name="test-index", schema=SCHEMA, timeout=12)

    assert clock.sleeps == [5.0, 5.0, 2.0]


@respx.mock
def test_sync_no_timeout_sleeps_full_interval(indexes: Indexes, clock: FakeClock) -> None:
    routes = _mock_never_ready()
    routes["describe"].mock(
        side_effect=[
            httpx.Response(200, json=make_index_response(status=NOT_READY)),
            httpx.Response(200, json=make_index_response(status={"ready": True, "state": "Ready"})),
        ]
    )

    indexes.create(name="test-index", schema=SCHEMA)

    assert clock.sleeps == [5.0]
