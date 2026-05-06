"""Unit tests for require_valid_resource_name validation helper."""

from __future__ import annotations

import pytest

from pinecone._internal.validation import require_valid_resource_name
from pinecone.errors.exceptions import ValidationError


def test_valid_name_lowercase_alphanumeric() -> None:
    require_valid_resource_name("name", "mycollection")


def test_valid_name_with_hyphens() -> None:
    require_valid_resource_name("name", "my-collection-2025")


def test_valid_name_max_length() -> None:
    require_valid_resource_name("name", "a" * 45)


def test_valid_name_single_char() -> None:
    require_valid_resource_name("name", "a")


def test_empty_string_raises() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        require_valid_resource_name("name", "")


def test_whitespace_only_raises() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        require_valid_resource_name("name", "   ")


def test_too_long_raises() -> None:
    with pytest.raises(ValidationError, match="too long"):
        require_valid_resource_name("name", "a" * 46)


def test_leading_hyphen_raises() -> None:
    with pytest.raises(ValidationError, match="must not start with a hyphen"):
        require_valid_resource_name("name", "-leading")


def test_trailing_hyphen_raises() -> None:
    with pytest.raises(ValidationError, match="must not end with a hyphen"):
        require_valid_resource_name("name", "trailing-")


def test_uppercase_raises() -> None:
    with pytest.raises(ValidationError, match="invalid characters"):
        require_valid_resource_name("name", "MY_COLLECTION")


def test_underscore_raises() -> None:
    with pytest.raises(ValidationError, match="invalid characters"):
        require_valid_resource_name("name", "under_score")


def test_at_symbol_raises() -> None:
    with pytest.raises(ValidationError, match="invalid characters"):
        require_valid_resource_name("name", "test@name")


def test_space_raises() -> None:
    with pytest.raises(ValidationError, match="invalid characters"):
        require_valid_resource_name("name", "my collection")


def test_error_message_includes_param_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        require_valid_resource_name("collection_name", "INVALID!")
    assert "collection_name" in str(exc_info.value)
