"""Tests for PineconeConfig __repr__ masking."""

from __future__ import annotations

from pinecone._internal.config import PineconeConfig


def test_repr_masks_full_api_key() -> None:
    config = PineconeConfig(api_key="pcsk_abc123def456")
    result = repr(config)
    assert "pcsk_abc123def456" not in result
    assert "...f456" in result


def test_repr_masks_short_api_key() -> None:
    config = PineconeConfig(api_key="abcd")
    result = repr(config)
    assert "abcd" not in result or "...abcd" in result
    assert "...abcd" in result


def test_repr_masks_very_short_api_key() -> None:
    config = PineconeConfig(api_key="ab")
    result = repr(config)
    assert "api_key='***'" in result


def test_repr_masks_empty_api_key() -> None:
    config = PineconeConfig(api_key="")
    result = repr(config)
    assert "api_key='***'" in result


def test_repr_includes_other_fields() -> None:
    config = PineconeConfig(
        api_key="pcsk_secret_key_12345",
        host="https://api.pinecone.io",
        timeout=60.0,
        source_tag="my_app",
    )
    result = repr(config)
    assert "host='https://api.pinecone.io'" in result
    assert "timeout=60.0" in result
    assert "source_tag='my_app'" in result
    assert "pcsk_secret_key_12345" not in result
