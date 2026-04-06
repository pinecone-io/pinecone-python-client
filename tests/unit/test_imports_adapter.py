"""Tests for the ImportsAdapter."""

from __future__ import annotations

import json

from pinecone._internal.adapters.imports_adapter import ImportsAdapter
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse


class TestToStartImportResponse:
    def test_to_start_import_response(self) -> None:
        data = json.dumps({"id": "abc-123"}).encode()
        result = ImportsAdapter.to_start_import_response(data)
        assert isinstance(result, StartImportResponse)
        assert result.id == "abc-123"


class TestToImportModel:
    def test_to_import_model(self) -> None:
        payload = {
            "id": "import-456",
            "uri": "s3://bucket/path/file.parquet",
            "status": "InProgress",
            "createdAt": "2025-01-15T10:30:00Z",
            "finishedAt": None,
            "percentComplete": 42.5,
            "recordsImported": 10000,
        }
        data = json.dumps(payload).encode()
        result = ImportsAdapter.to_import_model(data)
        assert isinstance(result, ImportModel)
        assert result.id == "import-456"
        assert result.uri == "s3://bucket/path/file.parquet"
        assert result.status == "InProgress"
        assert result.created_at == "2025-01-15T10:30:00Z"
        assert result.finished_at is None
        assert result.percent_complete == 42.5
        assert result.records_imported == 10000
        assert result.error is None

    def test_to_import_model_completed(self) -> None:
        payload = {
            "id": "import-789",
            "uri": "s3://bucket/data.parquet",
            "status": "Completed",
            "createdAt": "2025-01-15T10:30:00Z",
            "finishedAt": "2025-01-15T11:00:00Z",
            "percentComplete": 100.0,
            "recordsImported": 50000,
        }
        data = json.dumps(payload).encode()
        result = ImportsAdapter.to_import_model(data)
        assert result.status == "Completed"
        assert result.finished_at == "2025-01-15T11:00:00Z"
        assert result.percent_complete == 100.0
        assert result.records_imported == 50000


class TestToImportList:
    def test_to_import_list(self) -> None:
        payload = {
            "data": [
                {
                    "id": "import-1",
                    "uri": "s3://bucket/file1.parquet",
                    "status": "Completed",
                    "createdAt": "2025-01-15T10:00:00Z",
                },
                {
                    "id": "import-2",
                    "uri": "s3://bucket/file2.parquet",
                    "status": "InProgress",
                    "createdAt": "2025-01-15T11:00:00Z",
                },
            ],
            "pagination": {"next": "tok"},
        }
        data = json.dumps(payload).encode()
        result = ImportsAdapter.to_import_list(data)
        assert isinstance(result, ImportList)
        assert len(result) == 2
        items = list(result)
        assert items[0].id == "import-1"
        assert items[1].id == "import-2"

    def test_to_import_list_empty(self) -> None:
        data = json.dumps({"data": []}).encode()
        result = ImportsAdapter.to_import_list(data)
        assert isinstance(result, ImportList)
        assert len(result) == 0
