"""Tests for bulk import response models."""

from __future__ import annotations

import msgspec

from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse


class TestImportModelRequiredFields:
    def test_import_model_required_fields(self) -> None:
        model = ImportModel(
            id="imp-1",
            uri="s3://bucket/data.parquet",
            status="Pending",
            created_at="2025-01-01T00:00:00Z",
        )
        assert model.id == "imp-1"
        assert model.uri == "s3://bucket/data.parquet"
        assert model.status == "Pending"
        assert model.created_at == "2025-01-01T00:00:00Z"
        assert model.finished_at is None
        assert model.percent_complete is None
        assert model.records_imported is None
        assert model.error is None


class TestImportModelAllFields:
    def test_import_model_all_fields(self) -> None:
        model = ImportModel(
            id="imp-2",
            uri="s3://bucket/data.parquet",
            status="Completed",
            created_at="2025-01-01T00:00:00Z",
            finished_at="2025-01-01T01:00:00Z",
            percent_complete=100.0,
            records_imported=50000,
            error=None,
        )
        assert model.finished_at == "2025-01-01T01:00:00Z"
        assert model.percent_complete == 100.0
        assert model.records_imported == 50000
        assert model.error is None


class TestImportModelBracketAccess:
    def test_import_model_bracket_access(self) -> None:
        model = ImportModel(
            id="imp-3",
            uri="s3://bucket/data.parquet",
            status="InProgress",
            created_at="2025-01-01T00:00:00Z",
        )
        assert model["id"] == "imp-3"
        assert model["status"] == "InProgress"


class TestStartImportResponse:
    def test_start_import_response(self) -> None:
        resp = StartImportResponse(id="101")
        assert resp.id == "101"
        assert resp["id"] == "101"


class TestImportListIteration:
    def test_import_list_iteration(self) -> None:
        models = [
            ImportModel(
                id=f"imp-{i}",
                uri=f"s3://bucket/data{i}.parquet",
                status="Pending",
                created_at="2025-01-01T00:00:00Z",
            )
            for i in range(3)
        ]
        import_list = ImportList(models)
        assert len(import_list) == 3
        assert import_list[0].id == "imp-0"
        collected = list(import_list)
        assert len(collected) == 3
        assert collected[2].id == "imp-2"


class TestImportModelCamelCaseDecode:
    def test_import_model_camel_case_decode(self) -> None:
        raw = (
            b'{"id": "1", "uri": "s3://b", "status": "Pending",'
            b' "createdAt": "2025-01-01T00:00:00Z", "percentComplete": 42.0}'
        )
        model = msgspec.json.decode(raw, type=ImportModel)
        assert model.id == "1"
        assert model.uri == "s3://b"
        assert model.status == "Pending"
        assert model.created_at == "2025-01-01T00:00:00Z"
        assert model.percent_complete == 42.0
