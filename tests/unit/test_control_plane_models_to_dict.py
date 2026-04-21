"""Tests for to_dict() on control-plane model classes."""

from __future__ import annotations

from pinecone.models.backups.model import CreateIndexFromBackupResponse
from pinecone.models.collections.model import CollectionModel
from pinecone.models.imports.model import ImportModel, StartImportResponse
from pinecone.models.response_info import ResponseInfo


class TestCollectionModelToDict:
    def test_collection_model_to_dict_required(self) -> None:
        col = CollectionModel(name="col1", status="Ready", environment="us-east1-gcp")
        result = col.to_dict()
        assert isinstance(result, dict)
        assert result["name"] == "col1"
        assert result["status"] == "Ready"
        assert result["environment"] == "us-east1-gcp"
        assert "size" in result
        assert "dimension" in result
        assert "vector_count" in result

    def test_collection_model_to_dict_optional_none(self) -> None:
        col = CollectionModel(name="col1", status="Ready", environment="us-east1-gcp")
        result = col.to_dict()
        assert result["size"] is None
        assert result["dimension"] is None
        assert result["vector_count"] is None

    def test_collection_model_to_dict_with_optionals(self) -> None:
        col = CollectionModel(
            name="col1",
            status="Ready",
            environment="us-east1-gcp",
            size=1024,
            dimension=128,
            vector_count=500,
        )
        result = col.to_dict()
        assert result["size"] == 1024
        assert result["dimension"] == 128
        assert result["vector_count"] == 500


class TestImportModelToDict:
    def test_import_model_to_dict(self) -> None:
        model = ImportModel(
            id="import-1",
            uri="s3://bucket/data",
            status="Completed",
            created_at="2025-01-01T00:00:00Z",
        )
        result = model.to_dict()
        assert isinstance(result, dict)
        assert result["id"] == "import-1"
        assert result["uri"] == "s3://bucket/data"
        assert result["status"] == "Completed"
        assert result["created_at"] == "2025-01-01T00:00:00Z"

    def test_import_model_to_dict_optional_none(self) -> None:
        model = ImportModel(
            id="import-1",
            uri="s3://bucket/data",
            status="Pending",
            created_at="2025-01-01T00:00:00Z",
        )
        result = model.to_dict()
        assert result["finished_at"] is None
        assert result["percent_complete"] is None
        assert result["records_imported"] is None
        assert result["error"] is None


class TestStartImportResponseToDict:
    def test_start_import_response_to_dict(self) -> None:
        resp = StartImportResponse(id="import-42")
        result = resp.to_dict()
        assert isinstance(result, dict)
        assert result["id"] == "import-42"
        assert list(result.keys()) == ["id"]


class TestCreateIndexFromBackupResponseToDict:
    def test_create_index_from_backup_response_to_dict(self) -> None:
        resp = CreateIndexFromBackupResponse(restore_job_id="rj1", index_id="idx1")
        result = resp.to_dict()
        assert isinstance(result, dict)
        assert result["restore_job_id"] == "rj1"
        assert result["index_id"] == "idx1"

    def test_create_index_from_backup_response_keys(self) -> None:
        resp = CreateIndexFromBackupResponse(restore_job_id="rj1", index_id="idx1")
        result = resp.to_dict()
        assert set(result.keys()) == {"restore_job_id", "index_id"}


class TestResponseInfoToDict:
    def test_response_info_to_dict(self) -> None:
        info = ResponseInfo(request_id="req-123", lsn_reconciled=42)
        result = info.to_dict()
        assert isinstance(result, dict)
        assert result["request_id"] == "req-123"
        assert result["lsn_reconciled"] == 42

    def test_response_info_to_dict_all_none(self) -> None:
        info = ResponseInfo()
        result = info.to_dict()
        assert isinstance(result, dict)
        assert result["request_id"] is None
        assert result["lsn_reconciled"] is None
        assert result["lsn_committed"] is None


class TestToDictIsPureRead:
    def test_collection_model_to_dict_is_pure_read(self) -> None:
        col = CollectionModel(name="col1", status="Ready", environment="us-east1-gcp")
        result = col.to_dict()
        result["name"] = "mutated"
        assert col.name == "col1"

    def test_create_index_from_backup_response_to_dict_is_pure_read(self) -> None:
        resp = CreateIndexFromBackupResponse(restore_job_id="rj1", index_id="idx1")
        result = resp.to_dict()
        result["restore_job_id"] = "mutated"
        assert resp.restore_job_id == "rj1"
