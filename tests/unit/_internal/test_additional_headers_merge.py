"""``additional_headers`` replaces the SDK's own headers whatever the key's case."""

from __future__ import annotations

import httpx
import pytest

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient, _build_headers


def _config(**extra: str) -> PineconeConfig:
    return PineconeConfig(api_key="k", additional_headers=dict(extra))


def _values(headers: httpx.Headers, name: str) -> list[str]:
    return headers.get_list(name)


@pytest.mark.parametrize(
    "key",
    ["X-Pinecone-Api-Version", "X-Pinecone-API-Version", "x-pinecone-api-version"],
)
def test_build_headers_caller_version_replaces_sdk_version(key: str) -> None:
    headers = _build_headers(_config(**{key: "2025-10"}), "2026-07")

    versions = [v for k, v in headers.items() if k.lower() == "x-pinecone-api-version"]
    assert versions == ["2025-10"]


def test_build_headers_without_override_keeps_sdk_version() -> None:
    headers = _build_headers(_config(**{"X-Request-Source": "nightly"}), "2026-07")

    assert headers["X-Pinecone-Api-Version"] == "2026-07"
    assert headers["X-Request-Source"] == "nightly"


@pytest.mark.parametrize(
    ("key", "sdk_key"),
    [("USER-AGENT", "User-Agent"), ("api-key", "Api-Key")],
)
def test_build_headers_other_sdk_headers_are_replaced_case_insensitively(
    key: str, sdk_key: str
) -> None:
    headers = _build_headers(_config(**{key: "mine"}), "2026-07")

    matches = [v for k, v in headers.items() if k.lower() == sdk_key.lower()]
    assert matches == ["mine"]


def test_sync_client_sends_a_single_version_value() -> None:
    client = HTTPClient(_config(**{"X-Pinecone-API-Version": "2025-10"}), "2026-07")
    try:
        request = client._client.build_request("GET", "https://example.test/indexes")
        assert _values(request.headers, "x-pinecone-api-version") == ["2025-10"]
    finally:
        client.close()


async def test_async_client_sends_a_single_version_value() -> None:
    client = AsyncHTTPClient(_config(**{"X-Pinecone-API-Version": "2025-10"}), "2026-07")
    try:
        request = client._ensure_client().build_request("GET", "https://example.test/indexes")
        assert _values(request.headers, "x-pinecone-api-version") == ["2025-10"]
    finally:
        await client.close()
