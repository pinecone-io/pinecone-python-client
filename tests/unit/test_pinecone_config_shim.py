"""Unit tests for pinecone.config.pinecone_config backcompat shim."""

from __future__ import annotations

import logging

import pytest

from pinecone.config.config import Config
from pinecone.config.pinecone_config import PineconeConfig


def test_build_with_explicit_api_key_and_host_returns_config() -> None:
    result = PineconeConfig.build(api_key="sk-test", host="https://custom.example")
    assert isinstance(result, Config)
    assert result.api_key == "sk-test"
    assert result.host == "https://custom.example"
    assert result.additional_headers == {}


def test_build_without_host_falls_back_to_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "https://env.example")
    result = PineconeConfig.build(api_key="k")
    assert result.host == "https://env.example"


def test_build_without_host_or_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINECONE_CONTROLLER_HOST", raising=False)
    result = PineconeConfig.build(api_key="k")
    assert result.host == "https://api.pinecone.io"


def test_build_without_additional_headers_uses_empty_dict_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PINECONE_ADDITIONAL_HEADERS", raising=False)
    result = PineconeConfig.build(api_key="k")
    assert result.additional_headers == {}


def test_build_parses_additional_headers_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '{"X-Team": "core"}')
    result = PineconeConfig.build(api_key="k")
    assert result.additional_headers == {"X-Team": "core"}


def test_build_ignores_invalid_json_in_additional_headers_env(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", "not-json")
    with caplog.at_level(logging.WARNING):
        result = PineconeConfig.build(api_key="k")
    assert result.additional_headers == {}
    assert "PINECONE_ADDITIONAL_HEADERS" in caplog.text


def test_build_ignores_non_dict_json_in_additional_headers_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '["a", "b"]')
    result = PineconeConfig.build(api_key="k")
    assert result.additional_headers == {}


def test_build_explicit_additional_headers_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '{"X-From-Env": "yes"}')
    result = PineconeConfig.build(api_key="k", additional_headers={"X-Caller": "test"})
    assert result.additional_headers == {"X-Caller": "test"}


def test_build_filters_unknown_kwargs_and_passes_known_kwargs() -> None:
    result = PineconeConfig.build(
        api_key="k", source_tag="test-tag", not_a_field="ignored"
    )
    assert result.source_tag == "test-tag"
