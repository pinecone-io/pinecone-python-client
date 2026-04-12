"""Unit tests for PineconeConfig."""
from __future__ import annotations

from pinecone._internal.config import PineconeConfig


def test_repr_masks_api_key() -> None:
    config = PineconeConfig(api_key="abcdefgh")
    r = repr(config)
    assert "abcdefgh" not in r
    assert "efgh" in r


def test_repr_masks_proxy_headers() -> None:
    config = PineconeConfig(
        api_key="test-key",
        proxy_headers={"Proxy-Authorization": "Basic abc123"},
    )
    r = repr(config)
    assert "abc123" not in r
    assert "Proxy-Authorization" in r
    assert "***" in r


def test_repr_masks_additional_headers_auth() -> None:
    config = PineconeConfig(
        api_key="test-key",
        additional_headers={"Authorization": "Bearer secret", "X-Custom": "visible"},
    )
    r = repr(config)
    assert "secret" not in r
    assert "visible" in r
    assert "Authorization" in r
    assert "***" in r


def test_repr_masks_case_insensitive() -> None:
    config = PineconeConfig(
        api_key="test-key",
        additional_headers={"AUTHORIZATION": "Bearer secret"},
    )
    r = repr(config)
    assert "secret" not in r
    assert "***" in r


def test_repr_non_sensitive_headers_visible() -> None:
    config = PineconeConfig(
        api_key="test-key",
        additional_headers={"X-Custom-Header": "my-value"},
        proxy_headers={"X-Proxy-Extra": "proxy-value"},
    )
    r = repr(config)
    assert "my-value" in r
    assert "proxy-value" in r
