"""Tests for PineconeValueError, PineconeTypeError, and ValidationError alias."""

from __future__ import annotations

import pytest

from pinecone.errors.exceptions import (
    PineconeError,
    PineconeTypeError,
    PineconeValueError,
    ValidationError,
)


class TestPineconeValueError:
    def test_pinecone_value_error_inherits_pinecone_error(self) -> None:
        assert issubclass(PineconeValueError, PineconeError)

    def test_pinecone_value_error_inherits_value_error(self) -> None:
        assert issubclass(PineconeValueError, ValueError)

    def test_value_error_path_stored(self) -> None:
        e = PineconeValueError("bad", path="vectors[0].id")
        assert e.path == "vectors[0].id"

    def test_value_error_path_none_by_default(self) -> None:
        e = PineconeValueError("bad")
        assert e.path is None

    def test_catch_with_value_error(self) -> None:
        with pytest.raises(ValueError, match="bad"):
            raise PineconeValueError("bad")

    def test_catch_with_pinecone_error(self) -> None:
        with pytest.raises(PineconeError):
            raise PineconeValueError("bad")


class TestPineconeTypeError:
    def test_pinecone_type_error_inherits_pinecone_error(self) -> None:
        assert issubclass(PineconeTypeError, PineconeError)

    def test_pinecone_type_error_inherits_type_error(self) -> None:
        assert issubclass(PineconeTypeError, TypeError)

    def test_type_error_path_stored(self) -> None:
        e = PineconeTypeError("bad", path="metadata.key")
        assert e.path == "metadata.key"

    def test_catch_type_error_with_type_error(self) -> None:
        with pytest.raises(TypeError):
            raise PineconeTypeError("bad")


class TestValidationErrorAlias:
    def test_validation_error_is_pinecone_value_error(self) -> None:
        assert ValidationError is PineconeValueError
