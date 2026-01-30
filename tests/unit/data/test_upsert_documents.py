"""Tests for upsert_documents functionality."""

import pytest
from unittest.mock import MagicMock, patch

from pinecone.db_data.dataclasses import UpsertResponse
from pinecone.core.openapi.db_data.model.document_upsert_request import DocumentUpsertRequest


class TestDocumentUpsertRequest:
    """Tests for the DocumentUpsertRequest model."""

    def test_request_creation_with_documents(self):
        """Test creating a DocumentUpsertRequest with documents."""
        documents = [
            {"_id": "doc1", "title": "Test Title", "embedding": [0.1, 0.2, 0.3]},
            {"_id": "doc2", "title": "Another Title", "embedding": [0.4, 0.5, 0.6]},
        ]
        request = DocumentUpsertRequest(value=documents)
        assert request.value == documents

    def test_request_with_single_document(self):
        """Test creating a DocumentUpsertRequest with a single document."""
        documents = [{"_id": "doc1", "title": "Test", "embedding": [0.1, 0.2]}]
        request = DocumentUpsertRequest(value=documents)
        assert len(request.value) == 1
        assert request.value[0]["_id"] == "doc1"

    def test_request_with_various_field_types(self):
        """Test request with various field types in documents."""
        documents = [
            {
                "_id": "doc1",
                "title": "Test",
                "year": 2020,
                "rating": 8.5,
                "active": True,
                "tags": ["action", "comedy"],
                "embedding": [0.1, 0.2, 0.3],
            }
        ]
        request = DocumentUpsertRequest(value=documents)
        assert request.value[0]["year"] == 2020
        assert request.value[0]["rating"] == 8.5
        assert request.value[0]["active"] is True
        assert request.value[0]["tags"] == ["action", "comedy"]


class TestUpsertDocumentsValidation:
    """Tests for upsert_documents parameter validation."""

    def test_namespace_required(self):
        """Test that namespace is required."""
        from pinecone.db_data.index import Index

        # Create a mock index
        with patch.object(Index, "__init__", lambda self, *args, **kwargs: None):
            index = Index.__new__(Index)
            index._document_api = None

            # Test with None namespace
            with pytest.raises(ValueError, match="Namespace is required"):
                index.upsert_documents(namespace=None, documents=[{"_id": "1"}])

    def test_documents_required(self):
        """Test that documents list is required and cannot be empty."""
        from pinecone.db_data.index import Index

        with patch.object(Index, "__init__", lambda self, *args, **kwargs: None):
            index = Index.__new__(Index)
            index._document_api = None

            # Test with empty documents
            with pytest.raises(ValueError, match="At least one document is required"):
                index.upsert_documents(namespace="test", documents=[])


class TestUpsertDocumentsResponse:
    """Tests for UpsertResponse from upsert_documents."""

    def test_upsert_response_with_count(self):
        """Test UpsertResponse with upserted_count."""
        from pinecone.utils.response_info import extract_response_info

        response = UpsertResponse(upserted_count=5, _response_info=extract_response_info({}))
        assert response.upserted_count == 5

    def test_upsert_response_access(self):
        """Test accessing UpsertResponse fields."""
        from pinecone.utils.response_info import extract_response_info

        response = UpsertResponse(upserted_count=10, _response_info=extract_response_info({}))
        assert response.upserted_count == 10
        assert response["upserted_count"] == 10


class TestUpsertDocumentsIntegration:
    """Integration-style tests for upsert_documents with mocked API."""

    def test_upsert_documents_calls_api(self):
        """Test that upsert_documents correctly calls the document API."""
        from pinecone.db_data.index import Index

        with patch.object(Index, "__init__", lambda self, *args, **kwargs: None):
            index = Index.__new__(Index)

            # Mock the document_api
            mock_api = MagicMock()
            mock_response = MagicMock()
            mock_response.upserted_count = 2
            mock_response._response_info = None
            mock_api.upsert_documents.return_value = mock_response

            # Set up the mock
            index._document_api = mock_api

            # Call the method
            result = index.upsert_documents(
                namespace="test-namespace",
                documents=[{"_id": "doc1", "title": "Test 1"}, {"_id": "doc2", "title": "Test 2"}],
            )

            # Verify API was called
            mock_api.upsert_documents.assert_called_once()
            call_args = mock_api.upsert_documents.call_args
            assert call_args[0][0] == "test-namespace"
            assert isinstance(call_args[0][1], DocumentUpsertRequest)

            # Verify response
            assert result.upserted_count == 2

    def test_upsert_documents_uses_document_count_as_fallback(self):
        """Test fallback to document count when server doesn't return count."""
        from pinecone.db_data.index import Index

        with patch.object(Index, "__init__", lambda self, *args, **kwargs: None):
            index = Index.__new__(Index)

            # Mock the document_api with no upserted_count
            mock_api = MagicMock()
            mock_response = MagicMock()
            mock_response.upserted_count = None
            mock_response._response_info = None
            mock_api.upsert_documents.return_value = mock_response

            index._document_api = mock_api

            # Call with 3 documents
            result = index.upsert_documents(
                namespace="test",
                documents=[
                    {"_id": "1", "text": "a"},
                    {"_id": "2", "text": "b"},
                    {"_id": "3", "text": "c"},
                ],
            )

            # Should fall back to document count
            assert result.upserted_count == 3


class TestUpsertDocumentsAsyncio:
    """Tests for async upsert_documents."""

    @pytest.mark.asyncio
    async def test_async_upsert_documents_calls_api(self):
        """Test that async upsert_documents correctly calls the document API."""
        from pinecone.db_data.index_asyncio import _IndexAsyncio

        with patch.object(_IndexAsyncio, "__init__", lambda self, *args, **kwargs: None):
            index = _IndexAsyncio.__new__(_IndexAsyncio)

            # Mock the document_api
            mock_api = MagicMock()
            mock_response = MagicMock()
            mock_response.upserted_count = 2
            mock_response._response_info = None

            # Make upsert_documents return a coroutine
            async def mock_upsert(*args, **kwargs):
                return mock_response

            mock_api.upsert_documents = mock_upsert
            index._document_api = mock_api

            # Call the method
            result = await index.upsert_documents(
                namespace="test-namespace",
                documents=[{"_id": "doc1", "title": "Test 1"}, {"_id": "doc2", "title": "Test 2"}],
            )

            # Verify response
            assert result.upserted_count == 2

    @pytest.mark.asyncio
    async def test_async_namespace_required(self):
        """Test that namespace is required for async method."""
        from pinecone.db_data.index_asyncio import _IndexAsyncio

        with patch.object(_IndexAsyncio, "__init__", lambda self, *args, **kwargs: None):
            index = _IndexAsyncio.__new__(_IndexAsyncio)
            index._document_api = None

            with pytest.raises(ValueError, match="Namespace is required"):
                await index.upsert_documents(namespace=None, documents=[{"_id": "1"}])

    @pytest.mark.asyncio
    async def test_async_documents_required(self):
        """Test that documents list is required for async method."""
        from pinecone.db_data.index_asyncio import _IndexAsyncio

        with patch.object(_IndexAsyncio, "__init__", lambda self, *args, **kwargs: None):
            index = _IndexAsyncio.__new__(_IndexAsyncio)
            index._document_api = None

            with pytest.raises(ValueError, match="At least one document is required"):
                await index.upsert_documents(namespace="test", documents=[])
