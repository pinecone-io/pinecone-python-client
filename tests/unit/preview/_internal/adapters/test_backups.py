"""Unit tests for preview backup adapters."""

from __future__ import annotations

from pinecone.preview._internal.adapters.backups import (
    PreviewBackupModel,
    describe_backup_adapter,
    list_backups_adapter,
)

BACKUP_DICT: dict = {
    "backup_id": "bkp-001",
    "source_index_id": "idx-abc",
    "source_index_name": "my-index",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "created_at": "2024-01-15T10:30:00Z",
}


def test_describe_backup_adapter_parses_response() -> None:
    m = describe_backup_adapter.from_response(BACKUP_DICT)
    assert isinstance(m, PreviewBackupModel)
    assert m.backup_id == "bkp-001"
    assert m.source_index_id == "idx-abc"
    assert m.status == "Ready"


def test_describe_backup_adapter_ignores_unknown() -> None:
    payload = {**BACKUP_DICT, "metric": "cosine"}
    m = describe_backup_adapter.from_response(payload)
    assert m.backup_id == "bkp-001"


def test_list_backups_adapter_parses_empty() -> None:
    items, token = list_backups_adapter.from_response({"data": []})
    assert items == []
    assert token is None


def test_list_backups_adapter_parses_pagination_token() -> None:
    data = {"data": [BACKUP_DICT], "pagination": {"next": "abc"}}
    items, token = list_backups_adapter.from_response(data)
    assert len(items) == 1
    assert isinstance(items[0], PreviewBackupModel)
    assert token == "abc"


def test_list_backups_adapter_missing_pagination_key() -> None:
    data = {"data": [BACKUP_DICT]}
    items, token = list_backups_adapter.from_response(data)
    assert len(items) == 1
    assert token is None


def test_list_backups_adapter_null_next_token() -> None:
    data = {"data": [BACKUP_DICT], "pagination": {"next": None}}
    items, token = list_backups_adapter.from_response(data)
    assert len(items) == 1
    assert token is None
