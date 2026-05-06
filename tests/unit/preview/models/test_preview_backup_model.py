"""Unit tests for PreviewBackupModel — decode payloads for each status."""

from __future__ import annotations

import msgspec
import pytest

from pinecone.preview.models.backups import PreviewBackupModel

REQUIRED_FIELDS: dict = {
    "backup_id": "bkp-001",
    "source_index_id": "idx-abc",
    "source_index_name": "my-index",
    "cloud": "aws",
    "region": "us-east-1",
    "created_at": "2024-01-15T10:30:00Z",
}


def _make(status: str, **extra: object) -> PreviewBackupModel:
    return msgspec.convert({**REQUIRED_FIELDS, "status": status, **extra}, PreviewBackupModel)


def test_initializing_status_required_fields_only() -> None:
    m = _make("Initializing")
    assert m.backup_id == "bkp-001"
    assert m.source_index_id == "idx-abc"
    assert m.source_index_name == "my-index"
    assert m.status == "Initializing"
    assert m.cloud == "aws"
    assert m.region == "us-east-1"
    assert m.created_at == "2024-01-15T10:30:00Z"
    assert m.name is None
    assert m.description is None
    assert m.tags is None
    assert m.dimension is None
    assert m.schema is None
    assert m.record_count is None
    assert m.namespace_count is None
    assert m.size_bytes is None


def test_ready_status_with_optional_fields() -> None:
    m = _make(
        "Ready",
        name="nightly",
        description="Nightly backup",
        tags={"env": "prod"},
        dimension=1536,
        schema={"fields": [{"name": "text", "type": "text"}]},
        record_count=50000,
        namespace_count=3,
        size_bytes=104857600,
    )
    assert m.status == "Ready"
    assert m.name == "nightly"
    assert m.description == "Nightly backup"
    assert m.tags == {"env": "prod"}
    assert m.dimension == 1536
    assert isinstance(m.schema, dict)
    assert m.record_count == 50000
    assert m.namespace_count == 3
    assert m.size_bytes == 104857600


def test_failed_status() -> None:
    m = _make("Failed")
    assert m.status == "Failed"
    assert m.backup_id == "bkp-001"


def test_repr_format() -> None:
    m = _make("Ready")
    r = repr(m)
    assert r.startswith("PreviewBackupModel(")
    assert "backup_id=" in r
    assert "status=" in r
    assert "source_index_name=" in r
    assert "created_at=" in r
    assert r.endswith(")")


def test_ignores_unknown_fields() -> None:
    m = msgspec.convert(
        {**REQUIRED_FIELDS, "status": "Ready", "metric": "cosine", "unknown_future_field": 42},
        PreviewBackupModel,
    )
    assert m.backup_id == "bkp-001"


@pytest.mark.parametrize("status", ["Initializing", "Ready", "Failed"])
def test_all_statuses_decode(status: str) -> None:
    m = _make(status)
    assert m.status == status


def test_preview_backup_model_tags_non_string_values() -> None:
    raw = (
        b'{"backup_id":"bkp-1","source_index_id":"idx-abc",'
        b'"source_index_name":"my-index","status":"Ready",'
        b'"cloud":"aws","region":"us-east-1","created_at":"2026-01-01T00:00:00Z",'
        b'"tags":{"version":3,"env":"prod"}}'
    )
    model = msgspec.json.decode(raw, type=PreviewBackupModel)
    assert model.tags == {"version": 3, "env": "prod"}


def test_preview_backup_model_initialization_failed_status() -> None:
    raw = (
        b'{"backup_id":"bkp-2","source_index_id":"idx-abc",'
        b'"source_index_name":"my-index","status":"InitializationFailed",'
        b'"cloud":"aws","region":"us-east-1","created_at":"2026-01-01T00:00:00Z"}'
    )
    model = msgspec.json.decode(raw, type=PreviewBackupModel)
    assert model.status == "InitializationFailed"
