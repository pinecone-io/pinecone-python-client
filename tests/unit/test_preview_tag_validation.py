"""Unit tests for preview tag validation helper."""

from __future__ import annotations

import pytest

from pinecone.errors.exceptions import PineconeValueError
from pinecone.preview._internal.validation import validate_tags


def test_validate_tags_none_ok() -> None:
    validate_tags(None)


def test_validate_tags_valid_ok() -> None:
    validate_tags({"key-1": "val1", "key_2": "val2"})


def test_validate_tags_empty_value_ok() -> None:
    validate_tags({"key": ""})


def test_validate_tags_special_char_key_raises() -> None:
    with pytest.raises(PineconeValueError, match="invalid characters"):
        validate_tags({"bad-key!": "val"})


def test_validate_tags_non_ascii_key_raises() -> None:
    with pytest.raises(PineconeValueError, match="invalid characters"):
        validate_tags({"藏": "val"})


def test_validate_tags_non_ascii_value_raises() -> None:
    with pytest.raises(PineconeValueError, match="invalid characters"):
        validate_tags({"key": "日本語"})


def test_validate_tags_control_char_value_raises() -> None:
    with pytest.raises(PineconeValueError, match="invalid characters"):
        validate_tags({"key": "val\x00"})


def test_validate_tags_too_many_raises() -> None:
    tags = {f"key{i}": f"val{i}" for i in range(21)}
    with pytest.raises(PineconeValueError, match="maximum of 20"):
        validate_tags(tags)


def test_validate_tags_exactly_20_ok() -> None:
    tags = {f"key{i}": f"val{i}" for i in range(20)}
    validate_tags(tags)


def test_validate_tags_key_too_long_raises() -> None:
    with pytest.raises(PineconeValueError, match="80-character limit"):
        validate_tags({"a" * 81: "val"})


def test_validate_tags_key_exactly_80_ok() -> None:
    validate_tags({"a" * 80: "val"})


def test_validate_tags_value_too_long_raises() -> None:
    with pytest.raises(PineconeValueError, match="120-character limit"):
        validate_tags({"key": "v" * 121})


def test_validate_tags_value_exactly_120_ok() -> None:
    validate_tags({"key": "v" * 120})
