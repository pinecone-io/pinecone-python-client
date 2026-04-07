"""Tests for ResponseParsingError wrapping of msgspec decode failures."""

from __future__ import annotations

import json

import msgspec
import pytest

from pinecone._internal.adapters._decode import convert_response, decode_response
from pinecone._internal.adapters.collections_adapter import CollectionsAdapter
from pinecone.errors.exceptions import PineconeError, ResponseParsingError
from pinecone.models.collections.model import CollectionModel


class TestResponseParsingError:
    """ResponseParsingError is a PineconeError with a cause attribute."""

    def test_is_pinecone_error(self) -> None:
        exc = ResponseParsingError("boom")
        assert isinstance(exc, PineconeError)

    def test_stores_cause(self) -> None:
        cause = ValueError("original")
        exc = ResponseParsingError("wrapped", cause=cause)
        assert exc.cause is cause

    def test_cause_defaults_to_none(self) -> None:
        exc = ResponseParsingError("no cause")
        assert exc.cause is None

    def test_message(self) -> None:
        exc = ResponseParsingError("bad data")
        assert exc.message == "bad data"
        assert str(exc) == "bad data"


class TestDecodeResponse:
    """decode_response wraps msgspec errors in ResponseParsingError."""

    def test_invalid_json(self) -> None:
        with pytest.raises(ResponseParsingError) as exc_info:
            decode_response(b"not json", CollectionModel)
        assert "CollectionModel" in str(exc_info.value)
        assert exc_info.value.cause is not None

    def test_missing_required_field(self) -> None:
        data = json.dumps({"name": "x"}).encode()
        with pytest.raises(ResponseParsingError) as exc_info:
            decode_response(data, CollectionModel)
        assert exc_info.value.cause is not None

    def test_wrong_type(self) -> None:
        data = json.dumps({"name": 1, "status": "Ready", "environment": "us-east1"}).encode()
        with pytest.raises(ResponseParsingError) as exc_info:
            decode_response(data, CollectionModel)
        assert exc_info.value.cause is not None

    def test_valid_data_succeeds(self) -> None:
        data = json.dumps({"name": "test", "status": "Ready", "environment": "us-east1"}).encode()
        result = decode_response(data, CollectionModel)
        assert result.name == "test"
        assert result.status == "Ready"

    def test_caught_by_pinecone_error(self) -> None:
        """Users' except PineconeError blocks catch response parsing failures."""
        with pytest.raises(PineconeError):
            decode_response(b"not json", CollectionModel)

    def test_not_caught_by_validation_error(self) -> None:
        """ResponseParsingError is not a msgspec.ValidationError."""
        with pytest.raises(ResponseParsingError):
            decode_response(b"not json", CollectionModel)
        # Verify it does NOT match msgspec.ValidationError
        try:
            decode_response(b"not json", CollectionModel)
        except msgspec.ValidationError:
            pytest.fail("Should not be caught as msgspec.ValidationError")
        except ResponseParsingError:
            pass


class TestConvertResponse:
    """convert_response wraps msgspec.convert errors in ResponseParsingError."""

    def test_wrong_type(self) -> None:
        with pytest.raises(ResponseParsingError):
            convert_response({"name": 1, "status": "Ready", "environment": "us"}, CollectionModel)

    def test_valid_data_succeeds(self) -> None:
        result = convert_response(
            {"name": "test", "status": "Ready", "environment": "us-east1"},
            CollectionModel,
        )
        assert result.name == "test"


class TestAdapterWrapping:
    """Adapter methods raise ResponseParsingError, not raw msgspec errors."""

    def test_collections_adapter_invalid_json(self) -> None:
        with pytest.raises(ResponseParsingError):
            CollectionsAdapter.to_collection(b"not json")

    def test_collections_adapter_missing_field(self) -> None:
        data = json.dumps({"name": "x"}).encode()
        with pytest.raises(ResponseParsingError):
            CollectionsAdapter.to_collection(data)

    def test_collections_adapter_wrong_type(self) -> None:
        data = json.dumps({"name": 1, "status": "Ready", "environment": "us-east1"}).encode()
        with pytest.raises(ResponseParsingError):
            CollectionsAdapter.to_collection(data)

    def test_collections_adapter_list_invalid(self) -> None:
        with pytest.raises(ResponseParsingError):
            CollectionsAdapter.to_collection_list(b"not json")
