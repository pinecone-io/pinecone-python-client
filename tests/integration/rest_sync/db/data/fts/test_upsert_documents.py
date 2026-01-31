"""Integration tests for upsert_documents().

These tests verify the document upsert functionality for schema-based indexes.
"""

import pytest
import os
import uuid
from pinecone import text_query
from tests.integration.helpers import embedding_values, poll_until_lsn_reconciled

FTS_INDEX_DIMENSION = 8


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestUpsertDocuments:
    """Test upsert_documents() functionality."""

    def test_upsert_single_document(self, fts_index):
        """Test upserting a single document."""
        namespace = f"upsert-single-{str(uuid.uuid4())[:8]}"

        documents = [
            {
                "_id": "doc-1",
                "title": "Test Document",
                "description": "A test document for upserting.",
                "category": "test",
                "year": 2025,
                "embedding": embedding_values(FTS_INDEX_DIMENSION),
            }
        ]

        response = fts_index.upsert_documents(namespace=namespace, documents=documents)

        assert response.upserted_count == 1

    def test_upsert_multiple_documents(self, fts_index):
        """Test upserting multiple documents in a batch."""
        namespace = f"upsert-batch-{str(uuid.uuid4())[:8]}"

        documents = [
            {
                "_id": f"doc-{i}",
                "title": f"Document {i}",
                "description": f"Description for document {i}.",
                "category": "batch",
                "year": 2020 + i,
                "embedding": embedding_values(FTS_INDEX_DIMENSION),
            }
            for i in range(5)
        ]

        response = fts_index.upsert_documents(namespace=namespace, documents=documents)

        assert response.upserted_count == 5

    def test_upsert_and_search(self, fts_index):
        """Test that upserted documents are searchable."""
        namespace = f"upsert-search-{str(uuid.uuid4())[:8]}"

        documents = [
            {
                "_id": "searchable-doc",
                "title": "Unique Searchable Title XYZ789",
                "description": "A unique description for testing.",
                "category": "searchable",
                "year": 2024,
                "embedding": embedding_values(FTS_INDEX_DIMENSION),
            }
        ]

        response = fts_index.upsert_documents(namespace=namespace, documents=documents)
        assert response.upserted_count == 1

        poll_until_lsn_reconciled(fts_index, response._response_info, namespace=namespace)

        results = fts_index.search_documents(
            namespace=namespace, score_by=text_query("title", "XYZ789"), top_k=10
        )

        assert len(results.documents) == 1
        assert results.documents[0].id == "searchable-doc"
        assert "XYZ789" in results.documents[0].title

    def test_upsert_update_existing_document(self, fts_index):
        """Test that upserting with same _id updates the document."""
        namespace = f"upsert-update-{str(uuid.uuid4())[:8]}"

        original_doc = [
            {
                "_id": "update-doc",
                "title": "Original Title",
                "description": "Original description.",
                "category": "original",
                "year": 2020,
                "embedding": embedding_values(FTS_INDEX_DIMENSION),
            }
        ]

        response1 = fts_index.upsert_documents(namespace=namespace, documents=original_doc)
        assert response1.upserted_count == 1

        poll_until_lsn_reconciled(fts_index, response1._response_info, namespace=namespace)

        updated_doc = [
            {
                "_id": "update-doc",
                "title": "Updated Title QRS456",
                "description": "Updated description.",
                "category": "updated",
                "year": 2025,
                "embedding": embedding_values(FTS_INDEX_DIMENSION),
            }
        ]

        response2 = fts_index.upsert_documents(namespace=namespace, documents=updated_doc)
        assert response2.upserted_count == 1

        poll_until_lsn_reconciled(fts_index, response2._response_info, namespace=namespace)

        results = fts_index.search_documents(
            namespace=namespace, score_by=text_query("title", "QRS456"), top_k=10
        )

        assert len(results.documents) == 1
        assert results.documents[0].title == "Updated Title QRS456"
        assert results.documents[0].year == 2025

    def test_upsert_response_has_response_info(self, fts_index):
        """Test that upsert response includes response info with headers."""
        namespace = f"upsert-info-{str(uuid.uuid4())[:8]}"

        documents = [
            {
                "_id": "info-doc",
                "title": "Info Test",
                "description": "Testing response info.",
                "category": "info",
                "year": 2025,
                "embedding": embedding_values(FTS_INDEX_DIMENSION),
            }
        ]

        response = fts_index.upsert_documents(namespace=namespace, documents=documents)

        assert hasattr(response, "_response_info")
        assert response._response_info is not None


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestUpsertDocumentsEdgeCases:
    """Test edge cases for upsert_documents()."""

    def test_upsert_empty_documents_raises_error(self, fts_index):
        """Test that upserting empty document list raises an error."""
        namespace = f"upsert-empty-{str(uuid.uuid4())[:8]}"

        with pytest.raises(ValueError, match="At least one document is required"):
            fts_index.upsert_documents(namespace=namespace, documents=[])

    def test_upsert_without_namespace_raises_error(self, fts_index):
        """Test that upserting without namespace raises an error."""
        documents = [
            {
                "_id": "no-ns-doc",
                "title": "No Namespace",
                "description": "Should fail.",
                "category": "error",
                "year": 2025,
                "embedding": embedding_values(FTS_INDEX_DIMENSION),
            }
        ]

        with pytest.raises(ValueError, match="Namespace is required"):
            fts_index.upsert_documents(namespace=None, documents=documents)

    def test_upsert_large_batch(self, fts_index):
        """Test upserting a larger batch of documents."""
        namespace = f"upsert-large-{str(uuid.uuid4())[:8]}"

        documents = [
            {
                "_id": f"large-doc-{i}",
                "title": f"Large Batch Document {i}",
                "description": f"Description for large batch document {i}.",
                "category": "large",
                "year": 2020 + (i % 10),
                "embedding": embedding_values(FTS_INDEX_DIMENSION),
            }
            for i in range(50)
        ]

        response = fts_index.upsert_documents(namespace=namespace, documents=documents)

        assert response.upserted_count == 50
