"""Tests for deprecated symbol re-export paths in the top-level pinecone package."""

from __future__ import annotations

import importlib

import pytest

import pinecone
from pinecone import PineconeValueError
from pinecone.errors.exceptions import ValidationError as _CanonValidationError

_DEPRECATION_MATCH = "use PineconeValueError instead"


def _reset_validation_error_cache() -> None:
    pinecone.__dict__.pop("ValidationError", None)
    importlib.reload(pinecone)
    pinecone.__dict__.pop("ValidationError", None)


class TestValidationErrorDeprecation:
    def setup_method(self) -> None:
        _reset_validation_error_cache()

    def test_validation_error_deprec_warns(self) -> None:
        with pytest.warns(DeprecationWarning, match=_DEPRECATION_MATCH):
            from pinecone import ValidationError  # noqa: F401

    def test_validation_error_deprec_is_same_class(self) -> None:
        _reset_validation_error_cache()
        with pytest.warns(DeprecationWarning, match=_DEPRECATION_MATCH):
            from pinecone import ValidationError
        assert ValidationError is _CanonValidationError

    def test_validation_error_deprec_is_pinecone_value_error_subclass(self) -> None:
        _reset_validation_error_cache()
        with pytest.warns(DeprecationWarning, match=_DEPRECATION_MATCH):
            from pinecone import ValidationError
        assert issubclass(ValidationError, PineconeValueError)

    def test_validation_error_deprec_caches_on_module(self) -> None:
        _reset_validation_error_cache()
        with pytest.warns(DeprecationWarning, match=_DEPRECATION_MATCH):
            _ = pinecone.ValidationError
        assert "ValidationError" in pinecone.__dict__
