"""Unit tests for PreviewBackupModel and PreviewCreateBackupRequest."""

from __future__ import annotations

from unittest.mock import MagicMock

import msgspec

from pinecone.preview.models.backups import PreviewBackupModel, PreviewCreateBackupRequest

FULL_BACKUP_PAYLOAD: dict = {
    "backup_id": "bkp-001",
    "source_index_id": "idx-abc",
    "source_index_name": "my-index",
    "status": "Ready",
    "cloud": "aws",
    "region": "us-east-1",
    "created_at": "2024-01-15T10:30:00Z",
    "name": "nightly",
    "description": "Automated nightly backup",
    "tags": {"env": "prod", "team": "platform"},
    "dimension": 1536,
    "schema": {"fields": [{"name": "text", "type": "text"}]},
    "record_count": 50000,
    "namespace_count": 3,
    "size_bytes": 104857600,
}

MINIMAL_BACKUP_PAYLOAD: dict = {
    "backup_id": "bkp-001",
    "source_index_id": "idx-abc",
    "source_index_name": "my-index",
    "status": "Initializing",
    "cloud": "gcp",
    "region": "us-central1",
    "created_at": "2024-01-15T10:30:00Z",
}


def test_preview_backup_model_decode_full() -> None:
    m = msgspec.convert(FULL_BACKUP_PAYLOAD, PreviewBackupModel)
    assert m.backup_id == "bkp-001"
    assert m.source_index_id == "idx-abc"
    assert m.source_index_name == "my-index"
    assert m.status == "Ready"
    assert m.cloud == "aws"
    assert m.region == "us-east-1"
    assert m.created_at == "2024-01-15T10:30:00Z"
    assert m.name == "nightly"
    assert m.description == "Automated nightly backup"
    assert m.tags == {"env": "prod", "team": "platform"}
    assert m.dimension == 1536
    assert isinstance(m.schema, dict)
    assert m.record_count == 50000
    assert m.namespace_count == 3
    assert m.size_bytes == 104857600


def test_preview_backup_model_decode_minimal() -> None:
    m = msgspec.convert(MINIMAL_BACKUP_PAYLOAD, PreviewBackupModel)
    assert m.backup_id == "bkp-001"
    assert m.name is None
    assert m.description is None
    assert m.tags is None
    assert m.dimension is None
    assert m.schema is None
    assert m.record_count is None
    assert m.namespace_count is None
    assert m.size_bytes is None


def test_preview_backup_model_ignores_metric_field() -> None:
    payload = {**MINIMAL_BACKUP_PAYLOAD, "metric": "cosine"}
    m = msgspec.convert(payload, PreviewBackupModel)
    assert m.backup_id == "bkp-001"


def test_preview_backup_model_repr_includes_key_fields() -> None:
    m = msgspec.convert(FULL_BACKUP_PAYLOAD, PreviewBackupModel)
    r = repr(m)
    assert r.startswith("PreviewBackupModel(")
    assert "backup_id=" in r
    assert "status=" in r
    assert r.endswith(")")


def test_preview_backup_model_repr_pretty_cycle() -> None:
    m = msgspec.convert(MINIMAL_BACKUP_PAYLOAD, PreviewBackupModel)
    p = MagicMock()
    m._repr_pretty_(p, cycle=True)
    p.text.assert_called_with("PreviewBackupModel(...)")


def test_preview_backup_model_repr_html_contains_labels() -> None:
    m = msgspec.convert(FULL_BACKUP_PAYLOAD, PreviewBackupModel)
    html = m._repr_html_()
    assert "PreviewBackupModel" in html
    assert "Backup ID:" in html
    assert "Status:" in html
    assert "Source Index:" in html
    assert "Created:" in html


def test_preview_create_backup_request_all_optional() -> None:
    req = PreviewCreateBackupRequest()
    result = msgspec.to_builtins(req)
    assert result == {}


def test_preview_create_backup_request_with_name_only() -> None:
    req = PreviewCreateBackupRequest(name="nightly")
    result = msgspec.to_builtins(req)
    assert result == {"name": "nightly"}


def test_preview_backup_model_repr_pretty_non_cycle_emits_core_fields() -> None:
    m = msgspec.convert(MINIMAL_BACKUP_PAYLOAD, PreviewBackupModel)
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "PreviewBackupModel(" in emitted
    assert "backup_id='bkp-001'" in emitted
    assert "source_index_name='my-index'" in emitted
    assert "source_index_id='idx-abc'" in emitted
    assert "status='Initializing'" in emitted
    assert "cloud='gcp'" in emitted
    assert "region='us-central1'" in emitted
    assert "created_at='2024-01-15T10:30:00Z'" in emitted
    assert p.breakable.call_count >= 1
    assert "description=" not in emitted
    assert "dimension=" not in emitted
    assert "record_count=" not in emitted
    assert "namespace_count=" not in emitted
    assert "size_bytes=" not in emitted
    assert "tags=" not in emitted
    assert "schema=" not in emitted


def test_preview_backup_model_repr_pretty_non_cycle_includes_optional_fields_when_present() -> None:
    m = msgspec.convert(FULL_BACKUP_PAYLOAD, PreviewBackupModel)
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "name='nightly'" in emitted
    assert "description='Automated nightly backup'" in emitted
    assert "dimension=1536" in emitted
    assert "record_count=50000" in emitted
    assert "namespace_count=3" in emitted
    assert "size_bytes=104857600" in emitted
    assert "tags=" in emitted
    assert "schema=" in emitted


def test_preview_backup_model_repr_html_minimal_omits_optional_rows() -> None:
    m = msgspec.convert(MINIMAL_BACKUP_PAYLOAD, PreviewBackupModel)
    html = m._repr_html_()
    assert "Backup ID:" in html
    assert "Source Index:" in html
    assert "Status:" in html
    assert "Cloud:" in html
    assert "Region:" in html
    assert "Created:" in html
    assert "Name:" not in html
    assert "Description:" not in html
    assert "Dimension:" not in html
    assert "Records:" not in html
    assert "Namespaces:" not in html
    assert "Size:" not in html
    assert "Tags:" not in html


def test_preview_backup_model_repr_html_full_includes_optional_rows() -> None:
    m = msgspec.convert(FULL_BACKUP_PAYLOAD, PreviewBackupModel)
    html = m._repr_html_()
    assert "Name:" in html
    assert "Description:" in html
    assert "Dimension:" in html
    assert "Records:" in html
    assert "Namespaces:" in html
    assert "Size:" in html
    assert "Tags:" in html
    assert "env=prod" in html


def test_preview_create_backup_request_with_description_only() -> None:
    req = PreviewCreateBackupRequest(description="ctx only")
    result = msgspec.to_builtins(req)
    assert result == {"description": "ctx only"}


def test_preview_create_backup_request_with_both_fields() -> None:
    req = PreviewCreateBackupRequest(name="n", description="d")
    result = msgspec.to_builtins(req)
    assert result == {"name": "n", "description": "d"}
