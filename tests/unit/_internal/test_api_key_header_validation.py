"""An API key httpx cannot send is rejected locally, without echoing the key."""

from __future__ import annotations

import logging
import traceback

import httpx
import pytest

from pinecone import AsyncPinecone, Pinecone, PineconeConnectionError, PineconeValueError
from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient

FAKE_KEY = "fake-key-do-not-use"
BAD_KEYS = [
    "   ",
    "\t",
    f"{FAKE_KEY}\n",
    f"{FAKE_KEY}\r\n",
    f" {FAKE_KEY}",
    f"{FAKE_KEY} ",
    f"fake\x00{FAKE_KEY}",
    f"fake\x7f{FAKE_KEY}",
    f"caf\u00e9-{FAKE_KEY}",
]


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)


@pytest.mark.parametrize("key", BAD_KEYS)
def test_sync_client_rejects_unsendable_key(key: str) -> None:
    with pytest.raises(PineconeValueError) as info:
        Pinecone(api_key=key)
    assert FAKE_KEY not in str(info.value)
    assert repr(key) not in str(info.value)


@pytest.mark.parametrize("key", BAD_KEYS)
def test_async_client_rejects_unsendable_key(key: str) -> None:
    with pytest.raises(PineconeValueError) as info:
        AsyncPinecone(api_key=key)
    assert FAKE_KEY not in str(info.value)


@pytest.mark.parametrize("client_cls", [Pinecone, AsyncPinecone])
def test_env_var_key_is_checked_too(
    client_cls: type[Pinecone] | type[AsyncPinecone], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", f"{FAKE_KEY}\n")
    with pytest.raises(PineconeValueError):
        client_cls()


@pytest.mark.parametrize("key", [FAKE_KEY, "a b", "k\tv"])
def test_sendable_keys_are_accepted(key: str) -> None:
    Pinecone(api_key=key)
    AsyncPinecone(api_key=key)


def test_empty_key_still_reports_missing_key() -> None:
    with pytest.raises(PineconeValueError, match="No API key provided"):
        Pinecone(api_key="")


def _local_protocol_error() -> httpx.LocalProtocolError:
    return httpx.LocalProtocolError(f"Illegal header value b'{FAKE_KEY}\\n'")


def test_sync_local_protocol_error_is_not_retried_and_hides_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[int] = []

    def fail(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise _local_protocol_error()

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fail)
    client = HTTPClient(PineconeConfig(api_key=FAKE_KEY, host="http://127.0.0.1:1"), "2026-07")
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(PineconeConnectionError) as info:
            client.get("/indexes")
    assert len(calls) == 1
    assert FAKE_KEY not in str(info.value)
    assert FAKE_KEY not in "".join(traceback.format_exception(info.value))
    assert FAKE_KEY not in caplog.text


async def test_async_local_protocol_error_is_not_retried_and_hides_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[int] = []

    async def fail(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise _local_protocol_error()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fail)
    client = AsyncHTTPClient(PineconeConfig(api_key=FAKE_KEY, host="http://127.0.0.1:1"), "2026-07")
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(PineconeConnectionError) as info:
            await client.get("/indexes")
    assert len(calls) == 1
    assert FAKE_KEY not in str(info.value)
    assert FAKE_KEY not in "".join(traceback.format_exception(info.value))
    assert FAKE_KEY not in caplog.text


def test_index_clients_reject_unsendable_key() -> None:
    from pinecone import AsyncIndex, Index

    with pytest.raises(PineconeValueError):
        Index(api_key=f"{FAKE_KEY}\n", host="https://idx.example.test")
    with pytest.raises(PineconeValueError):
        AsyncIndex(api_key=f"{FAKE_KEY}\n", host="https://idx.example.test")


def test_config_alone_does_not_reject_unsendable_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", f"{FAKE_KEY}\n")
    PineconeConfig()


def test_bad_additional_header_value_hides_value(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise httpx.LocalProtocolError(f"Illegal header value b'{FAKE_KEY}\\n'")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fail)
    pc = Pinecone(
        api_key=FAKE_KEY, host="http://127.0.0.1:1", additional_headers={"X-Test": f"{FAKE_KEY}\n"}
    )
    with pytest.raises(PineconeConnectionError) as info:
        pc.indexes.list().to_list()
    assert FAKE_KEY not in "".join(traceback.format_exception(info.value))


async def test_async_bad_additional_header_value_hides_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    async def fail(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.LocalProtocolError(f"Illegal header value b'{FAKE_KEY}\\n'")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fail)
    async with AsyncPinecone(
        api_key=FAKE_KEY, host="http://127.0.0.1:1", additional_headers={"X-Test": f"{FAKE_KEY}\n"}
    ) as pc:
        with pytest.raises(PineconeConnectionError) as info:
            await pc.indexes.list().to_list()
    assert len(calls) == 1
    assert FAKE_KEY not in "".join(traceback.format_exception(info.value))


def test_illegal_header_value_is_scrubbed_from_the_whole_cause_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        inner = ValueError(f"Illegal header value b'{FAKE_KEY}\\n'")
        middle = RuntimeError(f"Illegal header value b'{FAKE_KEY}\\n'")
        middle.__cause__ = inner
        outer = httpx.LocalProtocolError(f"Illegal header value b'{FAKE_KEY}\\n'")
        outer.__cause__ = middle
        raise outer

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fail)
    pc = Pinecone(api_key=FAKE_KEY, host="http://127.0.0.1:1")
    with pytest.raises(PineconeConnectionError) as info:
        pc.indexes.list().to_list()
    assert FAKE_KEY not in "".join(traceback.format_exception(info.value))


@pytest.mark.parametrize(
    "key", [f" {FAKE_KEY}", f"{FAKE_KEY} ", f"\t{FAKE_KEY}\t", f" \t{FAKE_KEY} "]
)
def test_grpc_index_accepts_and_sends_edge_spaces_and_tabs(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pinecone._grpc as rust_grpc

    import pinecone.grpc as grpc_module

    sent: list[object] = []

    def channel(endpoint: str, api_key: str, *args: object, **kwargs: object) -> object:
        sent.append(api_key)
        return object()

    monkeypatch.setattr(rust_grpc, "GrpcChannel", channel)
    grpc_module.GrpcIndex(api_key=key, host="https://idx.example.test")
    assert sent == [key]


@pytest.mark.parametrize(
    "key",
    [
        "   ",
        " \t ",
        f"{FAKE_KEY}\n",
        f" {FAKE_KEY}\n",
        f"\n{FAKE_KEY}",
        f"fake\x00{FAKE_KEY}",
        f"café-{FAKE_KEY}",
    ],
)
def test_grpc_index_rejects_unsendable_key_before_allocating_resources(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pinecone.grpc as grpc_module

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("resources allocated before the key was checked")

    monkeypatch.setattr(grpc_module, "ThreadPoolExecutor", boom)
    with pytest.raises(PineconeValueError) as info:
        grpc_module.GrpcIndex(api_key=key, host="https://idx.example.test")
    assert FAKE_KEY not in str(info.value)
