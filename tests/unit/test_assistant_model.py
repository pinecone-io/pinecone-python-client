"""Tests for AssistantModel and AssistantFileModel dict-like access."""

import pytest

from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.model import AssistantModel


class TestAssistantModelDictAccess:
    def setup_method(self) -> None:
        self.model = AssistantModel(
            name="test-assistant",
            status="Ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            metadata={"key": "value"},
            instructions="Be helpful",
            host="https://example.com",
        )

    def test_getitem_valid_key(self) -> None:
        assert self.model["name"] == "test-assistant"
        assert self.model["status"] == "Ready"
        assert self.model["metadata"] == {"key": "value"}

    def test_getitem_optional_none(self) -> None:
        model = AssistantModel(
            name="test",
            status="Ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert model["metadata"] is None
        assert model["instructions"] is None
        assert model["host"] is None

    def test_getitem_invalid_key_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            self.model["nonexistent"]

    def test_contains_present_key(self) -> None:
        assert "name" in self.model
        assert "status" in self.model
        assert "metadata" in self.model
        assert "instructions" in self.model
        assert "host" in self.model
        assert "created_at" in self.model
        assert "updated_at" in self.model

    def test_contains_absent_key(self) -> None:
        assert "nonexistent" not in self.model
        assert "id" not in self.model
        assert "" not in self.model


class TestAssistantFileModelDictAccess:
    def setup_method(self) -> None:
        self.model = AssistantFileModel(
            name="test-file.pdf",
            id="file-123",
            metadata={"key": "value"},
            created_on="2024-01-01T00:00:00Z",
            updated_on="2024-01-01T00:00:00Z",
            status="Available",
            size=1024,
            multimodal=False,
            signed_url="https://example.com/file",
            content_hash="abc123",
        )

    def test_getitem_valid_key(self) -> None:
        assert self.model["name"] == "test-file.pdf"
        assert self.model["id"] == "file-123"
        assert self.model["size"] == 1024
        assert self.model["multimodal"] is False

    def test_getitem_optional_none(self) -> None:
        model = AssistantFileModel(name="test.pdf", id="file-456")
        assert model["metadata"] is None
        assert model["status"] is None
        assert model["signed_url"] is None

    def test_getitem_invalid_key_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            self.model["nonexistent"]

    def test_contains_present_key(self) -> None:
        assert "name" in self.model
        assert "id" in self.model
        assert "metadata" in self.model
        assert "status" in self.model
        assert "size" in self.model
        assert "signed_url" in self.model
        assert "content_hash" in self.model

    def test_contains_absent_key(self) -> None:
        assert "nonexistent" not in self.model
        assert "backup_id" not in self.model
        assert "" not in self.model
