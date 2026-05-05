"""Tests for backup and restore response models."""

from __future__ import annotations

import msgspec
import pytest

from pinecone.models.backups.list import BackupList, RestoreJobList
from pinecone.models.backups.model import (
    BackupModel,
    CreateIndexFromBackupResponse,
    RestoreJobModel,
)


def _make_backup(**overrides: object) -> BackupModel:
    defaults: dict[str, object] = {
        "backup_id": "bkp-123",
        "source_index_name": "my-index",
        "source_index_id": "idx-456",
        "status": "Ready",
        "cloud": "aws",
        "region": "us-east-1",
    }
    defaults.update(overrides)
    return BackupModel(**defaults)  # type: ignore[arg-type]


def _make_restore_job(**overrides: object) -> RestoreJobModel:
    defaults: dict[str, object] = {
        "restore_job_id": "rj-001",
        "backup_id": "bkp-123",
        "target_index_name": "restored-index",
        "target_index_id": "idx-789",
        "status": "Running",
        "created_at": "2025-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return RestoreJobModel(**defaults)  # type: ignore[arg-type]


class TestBackupModelRequiredFields:
    def test_backup_model_required_fields(self) -> None:
        backup = _make_backup()
        assert backup.backup_id == "bkp-123"
        assert backup.source_index_name == "my-index"
        assert backup.source_index_id == "idx-456"
        assert backup.status == "Ready"
        assert backup.cloud == "aws"
        assert backup.region == "us-east-1"
        assert backup.name is None
        assert backup.description is None
        assert backup.dimension is None
        assert backup.metric is None
        assert backup.record_count is None
        assert backup.namespace_count is None
        assert backup.size_bytes is None
        assert backup.tags is None
        assert backup.created_at is None


class TestBackupModelAllFields:
    def test_backup_model_all_fields(self) -> None:
        backup = _make_backup(
            name="daily-backup",
            description="Daily backup of production index",
            dimension=1536,
            metric="cosine",
            record_count=100000,
            namespace_count=5,
            size_bytes=52428800,
            tags={"env": "prod"},
            created_at="2025-01-15T10:30:00Z",
        )
        assert backup.name == "daily-backup"
        assert backup.description == "Daily backup of production index"
        assert backup.dimension == 1536
        assert backup.metric == "cosine"
        assert backup.record_count == 100000
        assert backup.namespace_count == 5
        assert backup.size_bytes == 52428800
        assert backup.tags == {"env": "prod"}
        assert backup.created_at == "2025-01-15T10:30:00Z"


class TestBackupModelBracketAccess:
    def test_backup_model_bracket_access(self) -> None:
        backup = _make_backup()
        assert backup["backup_id"] == "bkp-123"
        assert backup["status"] == "Ready"

    def test_backup_model_bracket_access_invalid_key(self) -> None:
        backup = _make_backup()
        with pytest.raises(KeyError, match="nonexistent"):
            backup["nonexistent"]


class TestBackupModelSchema:
    def test_backup_model_schema_decoded(self) -> None:
        """BackupModel must expose schema when returned by backend."""
        raw = (
            b'{"backup_id":"bkp-1","source_index_name":"my-index",'
            b'"source_index_id":"idx-abc","status":"Ready","cloud":"aws",'
            b'"region":"us-east-1",'
            b'"schema":{"fields":{"genre":{"filterable":true}}}}'
        )
        model = msgspec.json.decode(raw, type=BackupModel)
        assert model.schema is not None
        assert model.schema["fields"]["genre"]["filterable"] is True

    def test_backup_model_schema_absent(self) -> None:
        """BackupModel.schema is None when backend omits the field."""
        raw = (
            b'{"backup_id":"bkp-1","source_index_name":"my-index",'
            b'"source_index_id":"idx-abc","status":"Ready","cloud":"aws",'
            b'"region":"us-east-1"}'
        )
        model = msgspec.json.decode(raw, type=BackupModel)
        assert model.schema is None


class TestBackupModelJsonDecode:
    def test_backup_model_json_decode(self) -> None:
        payload = b"""{
            "backup_id": "bkp-abc",
            "source_index_name": "test-index",
            "source_index_id": "idx-def",
            "status": "Ready",
            "cloud": "gcp",
            "region": "us-central1",
            "name": "my-backup",
            "description": "Test backup",
            "dimension": 768,
            "metric": "dotproduct",
            "record_count": 5000,
            "namespace_count": 2,
            "size_bytes": 1048576,
            "tags": {"team": "ml"},
            "created_at": "2025-03-01T12:00:00Z"
        }"""
        backup = msgspec.json.decode(payload, type=BackupModel)
        assert backup.backup_id == "bkp-abc"
        assert backup.source_index_name == "test-index"
        assert backup.cloud == "gcp"
        assert backup.region == "us-central1"
        assert backup.name == "my-backup"
        assert backup.dimension == 768
        assert backup.tags == {"team": "ml"}


class TestRestoreJobModelRequiredFields:
    def test_restore_job_model_required_fields(self) -> None:
        job = _make_restore_job()
        assert job.restore_job_id == "rj-001"
        assert job.backup_id == "bkp-123"
        assert job.target_index_name == "restored-index"
        assert job.target_index_id == "idx-789"
        assert job.status == "Running"
        assert job.created_at == "2025-01-01T00:00:00Z"
        assert job.completed_at is None
        assert job.percent_complete is None


class TestRestoreJobModelCompleted:
    def test_restore_job_model_completed(self) -> None:
        job = _make_restore_job(
            status="Completed",
            completed_at="2025-01-01T01:00:00Z",
            percent_complete=100.0,
        )
        assert job.status == "Completed"
        assert job.completed_at == "2025-01-01T01:00:00Z"
        assert job.percent_complete == 100.0


class TestBackupListIteration:
    def test_backup_list_iteration(self) -> None:
        b1 = _make_backup(backup_id="bkp-1", name="first")
        b2 = _make_backup(backup_id="bkp-2", name=None)
        bl = BackupList([b1, b2])

        assert len(bl) == 2
        assert bl[0] is b1
        assert bl[1] is b2
        assert list(bl) == [b1, b2]
        assert bl.names() == ["first", "bkp-2"]


class TestRestoreJobListIteration:
    def test_restore_job_list_iteration(self) -> None:
        j1 = _make_restore_job(restore_job_id="rj-1")
        j2 = _make_restore_job(restore_job_id="rj-2")
        jl = RestoreJobList([j1, j2])

        assert len(jl) == 2
        assert list(jl) == [j1, j2]
        assert jl[0] is j1


class TestCreateFromBackupResponse:
    def test_create_from_backup_response(self) -> None:
        resp = CreateIndexFromBackupResponse(
            restore_job_id="rj-100",
            index_id="idx-new",
        )
        assert resp.restore_job_id == "rj-100"
        assert resp.index_id == "idx-new"
        assert resp["restore_job_id"] == "rj-100"
        assert resp["index_id"] == "idx-new"
