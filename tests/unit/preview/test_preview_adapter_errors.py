"""Tests that preview adapters wrap decode errors as ResponseParsingError."""

from __future__ import annotations

import httpx
import msgspec
import pytest

from pinecone.errors.exceptions import ResponseParsingError
from pinecone.preview._internal.adapters.backups import (
    PreviewDescribeBackupAdapter,
    PreviewListBackupsAdapter,
)
from pinecone.preview._internal.adapters.documents import PreviewDocumentsAdapter
from pinecone.preview._internal.adapters.indexes import (
    PreviewCreateIndexAdapter,
    PreviewDescribeIndexAdapter,
    PreviewListIndexesAdapter,
)

# ---------------------------------------------------------------------------
# Index adapters
# ---------------------------------------------------------------------------


def test_preview_adapter_wraps_validation_error_as_response_parsing_error() -> None:
    raw = b'{"name": 5, "host": "h", "status": {"ready": true, "state": "Ready"}, "schema": {"fields": {}}, "deployment": {"deployment_type": "managed", "environment": "e", "cloud": "aws", "region": "us-east-1"}, "deletion_protection": "disabled"}'
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewCreateIndexAdapter.from_response(raw)
    assert isinstance(exc_info.value.__cause__, msgspec.ValidationError)


def test_preview_adapter_wraps_decode_error_as_response_parsing_error() -> None:
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewCreateIndexAdapter.from_response(b"not json")
    assert isinstance(exc_info.value.__cause__, msgspec.DecodeError)


def test_preview_describe_index_adapter_wraps_validation_error() -> None:
    raw = b'{"name": 5, "host": "h", "status": {"ready": true, "state": "Ready"}, "schema": {"fields": {}}, "deployment": {"deployment_type": "managed", "environment": "e", "cloud": "aws", "region": "us-east-1"}, "deletion_protection": "disabled"}'
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewDescribeIndexAdapter.from_response(raw)
    assert isinstance(exc_info.value.__cause__, msgspec.ValidationError)


def test_preview_describe_index_adapter_wraps_decode_error() -> None:
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewDescribeIndexAdapter.from_response(b"not json")
    assert isinstance(exc_info.value.__cause__, msgspec.DecodeError)


def test_preview_list_indexes_adapter_wraps_decode_error() -> None:
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewListIndexesAdapter.from_response(b"not json")
    assert isinstance(exc_info.value.__cause__, msgspec.DecodeError)


# ---------------------------------------------------------------------------
# Backup adapters
# ---------------------------------------------------------------------------


def test_preview_backup_adapter_wraps_validation_error_as_response_parsing_error() -> None:
    # backup_id should be a string, not an integer
    raw = b'{"backup_id": 42, "source_index_id": "idx", "source_index_name": "n", "status": "Ready", "cloud": "aws", "region": "us-east-1", "created_at": "2024-01-01T00:00:00Z"}'
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewDescribeBackupAdapter.from_response(raw)
    assert isinstance(exc_info.value.__cause__, msgspec.ValidationError)


def test_preview_backup_adapter_wraps_decode_error_as_response_parsing_error() -> None:
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewDescribeBackupAdapter.from_response(b"not json")
    assert isinstance(exc_info.value.__cause__, msgspec.DecodeError)


def test_preview_list_backups_adapter_wraps_decode_error() -> None:
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewListBackupsAdapter.from_response(b"not json")
    assert isinstance(exc_info.value.__cause__, msgspec.DecodeError)


# ---------------------------------------------------------------------------
# Documents adapter
# ---------------------------------------------------------------------------


def test_preview_documents_search_adapter_wraps_decode_error() -> None:
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewDocumentsAdapter.to_search_response(httpx.Response(200, content=b"not json"))
    assert isinstance(exc_info.value.__cause__, msgspec.DecodeError)


def test_preview_documents_fetch_adapter_wraps_decode_error() -> None:
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewDocumentsAdapter.to_fetch_response(b"not json")
    assert isinstance(exc_info.value.__cause__, msgspec.DecodeError)


def test_preview_documents_search_adapter_wraps_validation_error() -> None:
    # usage.read_units must be an int, not a string
    raw = b'{"matches": [], "namespace": "ns", "usage": {"read_units": "bad"}}'
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewDocumentsAdapter.to_search_response(httpx.Response(200, content=raw))
    assert isinstance(exc_info.value.__cause__, msgspec.ValidationError)


def test_preview_documents_fetch_adapter_wraps_validation_error() -> None:
    raw = b'{"documents": [], "namespace": "ns", "usage": {"read_units": "bad"}}'
    with pytest.raises(ResponseParsingError) as exc_info:
        PreviewDocumentsAdapter.to_fetch_response(raw)
    assert isinstance(exc_info.value.__cause__, (msgspec.ValidationError, msgspec.DecodeError))
