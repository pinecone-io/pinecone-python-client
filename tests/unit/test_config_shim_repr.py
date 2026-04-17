"""Tests for Config shim __repr__ credential masking."""

from __future__ import annotations

from pinecone.config.config import Config


def test_repr_masks_api_key() -> None:
    cfg = Config(api_key="sk-supersecret12345")
    r = repr(cfg)
    assert "sk-supersecret12345" not in r
    assert "...2345" in r


def test_repr_masks_short_api_key() -> None:
    cfg = Config(api_key="abc")
    r = repr(cfg)
    assert "***" in r
    assert "abc" not in r


def test_repr_redacts_additional_headers() -> None:
    cfg = Config(additional_headers={"Authorization": "Bearer secret"})
    r = repr(cfg)
    assert "secret" not in r
    assert "***" in r


def test_repr_redacts_proxy_headers() -> None:
    cfg = Config(proxy_headers={"Proxy-Authorization": "Basic sekret"})
    r = repr(cfg)
    assert "sekret" not in r
    assert "***" in r


def test_repr_preserves_other_fields() -> None:
    cfg = Config(
        api_key="sk-supersecret12345",
        host="https://api.pinecone.io",
        source_tag="my-app",
        ssl_verify=True,
    )
    r = repr(cfg)
    assert "https://api.pinecone.io" in r
    assert "my-app" in r
    assert "ssl_verify=True" in r


def test_repr_none_proxy_headers() -> None:
    cfg = Config(proxy_headers=None)
    r = repr(cfg)
    assert "proxy_headers=None" in r
