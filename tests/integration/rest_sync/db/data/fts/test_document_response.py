"""Integration tests for Document and DocumentSearchResponse.

These tests verify the response structure and access patterns for document search results.
"""

import pytest
import os
from pinecone import text_query


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestDocumentAttributeAccess:
    """Test Document attribute access patterns."""

    def test_document_id_property(self, fts_index, seeded_fts_namespace):
        """Test accessing document id via property."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=1
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        # Access via property
        assert doc.id is not None
        assert isinstance(doc.id, str)

    def test_document_score_property(self, fts_index, seeded_fts_namespace):
        """Test accessing document score via property."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=1
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        # Access via property
        assert doc.score is not None
        assert isinstance(doc.score, (int, float))
        assert doc.score >= 0

    def test_document_dynamic_field_attribute_access(self, fts_index, seeded_fts_namespace):
        """Test accessing dynamic fields via attribute access."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["*"],
            top_k=1,
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        # Access dynamic fields via attribute
        assert doc.title is not None
        assert doc.category is not None
        assert doc.year is not None

    def test_document_missing_attribute_raises_error(self, fts_index, seeded_fts_namespace):
        """Test that accessing a non-existent attribute raises AttributeError."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=1
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        with pytest.raises(AttributeError):
            _ = doc.nonexistent_field


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestDocumentDictAccess:
    """Test Document dict-style access patterns."""

    def test_document_dict_access_id(self, fts_index, seeded_fts_namespace):
        """Test accessing id via dict-style access."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=1
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        # Access via dict syntax
        assert doc["id"] is not None
        assert doc["id"] == doc.id

    def test_document_dict_access_score(self, fts_index, seeded_fts_namespace):
        """Test accessing score via dict-style access."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=1
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        # Access via dict syntax
        assert doc["score"] is not None
        assert doc["score"] == doc.score

    def test_document_dict_access_dynamic_fields(self, fts_index, seeded_fts_namespace):
        """Test accessing dynamic fields via dict-style access."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["*"],
            top_k=1,
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        # Access dynamic fields via dict syntax
        assert doc["title"] is not None
        assert doc["category"] is not None
        assert doc["year"] is not None

    def test_document_missing_key_raises_error(self, fts_index, seeded_fts_namespace):
        """Test that accessing a non-existent key raises KeyError."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=1
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        with pytest.raises(KeyError):
            _ = doc["nonexistent_key"]


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestDocumentGetMethod:
    """Test Document get() method with defaults."""

    def test_document_get_existing_field(self, fts_index, seeded_fts_namespace):
        """Test get() method for existing field returns value."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["*"],
            top_k=1,
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        # get() returns value for existing field
        assert doc.get("id") == doc.id
        assert doc.get("score") == doc.score
        assert doc.get("title") == doc.title

    def test_document_get_missing_field_returns_none(self, fts_index, seeded_fts_namespace):
        """Test get() method for missing field returns None by default."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=1
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        # get() returns None for missing field
        assert doc.get("nonexistent_field") is None

    def test_document_get_missing_field_returns_default(self, fts_index, seeded_fts_namespace):
        """Test get() method for missing field returns provided default."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=1
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        # get() returns custom default for missing field
        assert doc.get("nonexistent_field", "default_value") == "default_value"
        assert doc.get("nonexistent_field", 42) == 42


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestDocumentContains:
    """Test Document __contains__ method for 'in' operator."""

    def test_document_contains_standard_fields(self, fts_index, seeded_fts_namespace):
        """Test 'in' operator for standard fields."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=1
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        assert "id" in doc
        assert "score" in doc

    def test_document_contains_dynamic_fields(self, fts_index, seeded_fts_namespace):
        """Test 'in' operator for dynamic fields."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["*"],
            top_k=1,
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        assert "title" in doc
        assert "category" in doc
        assert "year" in doc

    def test_document_not_contains_missing_field(self, fts_index, seeded_fts_namespace):
        """Test 'in' operator returns False for missing fields."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=1
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        assert "nonexistent_field" not in doc


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestDocumentIteration:
    """Test Document iteration and dict-like methods."""

    def test_document_keys(self, fts_index, seeded_fts_namespace):
        """Test Document.keys() method."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["*"],
            top_k=1,
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        keys = doc.keys()
        assert "id" in keys
        assert "score" in keys
        assert "title" in keys

    def test_document_values(self, fts_index, seeded_fts_namespace):
        """Test Document.values() method."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["*"],
            top_k=1,
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        values = doc.values()
        assert len(values) >= 2  # At least id and score

    def test_document_items(self, fts_index, seeded_fts_namespace):
        """Test Document.items() method."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["*"],
            top_k=1,
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        items = doc.items()
        items_dict = dict(items)
        assert "id" in items_dict
        assert "score" in items_dict

    def test_document_iteration(self, fts_index, seeded_fts_namespace):
        """Test iterating over Document yields field names."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["*"],
            top_k=1,
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        field_names = list(doc)
        assert "id" in field_names
        assert "score" in field_names

    def test_document_to_dict(self, fts_index, seeded_fts_namespace):
        """Test Document.to_dict() method."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["*"],
            top_k=1,
        )

        assert len(results.documents) >= 1
        doc = results.documents[0]

        doc_dict = doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert "id" in doc_dict
        assert "score" in doc_dict
        assert doc_dict["id"] == doc.id
        assert doc_dict["score"] == doc.score


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestDocumentSearchResponseStructure:
    """Test DocumentSearchResponse structure."""

    def test_response_has_documents_list(self, fts_index, seeded_fts_namespace):
        """Test that response has documents list."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=10
        )

        assert hasattr(results, "documents")
        assert isinstance(results.documents, list)

    def test_response_has_usage(self, fts_index, seeded_fts_namespace):
        """Test that response has usage information."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=10
        )

        assert hasattr(results, "usage")
        assert results.usage is not None

    def test_response_has_response_info(self, fts_index, seeded_fts_namespace):
        """Test that response has _response_info with headers."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "panther"), top_k=10
        )

        assert hasattr(results, "_response_info")
        assert results._response_info is not None
