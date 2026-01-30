"""Tests for Document and DocumentSearchResponse classes."""

import pytest
from pinecone.db_data.dataclasses import Document, DocumentSearchResponse


class TestDocument:
    """Tests for the Document class."""

    def test_basic_construction(self):
        """Test creating a document with id and score."""
        doc = Document(id="doc1", score=0.95)
        assert doc.id == "doc1"
        assert doc.score == 0.95

    def test_construction_with_fields(self):
        """Test creating a document with dynamic fields."""
        doc = Document(id="doc1", score=0.95, title="Test Title", year=2024)
        assert doc.id == "doc1"
        assert doc.score == 0.95
        assert doc.title == "Test Title"
        assert doc.year == 2024

    def test_attribute_access_dynamic_fields(self):
        """Test accessing dynamic fields via attributes."""
        doc = Document(id="doc1", score=0.8, title="Pink Panther", genre="Comedy")
        assert doc.title == "Pink Panther"
        assert doc.genre == "Comedy"

    def test_attribute_access_missing_field(self):
        """Test AttributeError for missing dynamic fields."""
        doc = Document(id="doc1", score=0.8)
        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            _ = doc.nonexistent

    def test_dict_access_standard_fields(self):
        """Test dict-style access to standard fields."""
        doc = Document(id="doc1", score=0.75)
        assert doc["id"] == "doc1"
        assert doc["score"] == 0.75

    def test_dict_access_dynamic_fields(self):
        """Test dict-style access to dynamic fields."""
        doc = Document(id="doc1", score=0.8, title="Test")
        assert doc["title"] == "Test"

    def test_dict_access_missing_field(self):
        """Test KeyError for missing fields."""
        doc = Document(id="doc1", score=0.8)
        with pytest.raises(KeyError):
            _ = doc["nonexistent"]

    def test_get_standard_fields(self):
        """Test get() method for standard fields."""
        doc = Document(id="doc1", score=0.9)
        assert doc.get("id") == "doc1"
        assert doc.get("score") == 0.9

    def test_get_dynamic_fields(self):
        """Test get() method for dynamic fields."""
        doc = Document(id="doc1", score=0.9, title="Test")
        assert doc.get("title") == "Test"

    def test_get_missing_field_returns_none(self):
        """Test get() returns None for missing fields."""
        doc = Document(id="doc1", score=0.9)
        assert doc.get("nonexistent") is None

    def test_get_missing_field_returns_default(self):
        """Test get() returns default value for missing fields."""
        doc = Document(id="doc1", score=0.9)
        assert doc.get("nonexistent", "N/A") == "N/A"
        assert doc.get("nonexistent", 0) == 0

    def test_keys(self):
        """Test keys() method."""
        doc = Document(id="doc1", score=0.9, title="Test", year=2024)
        keys = doc.keys()
        assert "id" in keys
        assert "score" in keys
        assert "title" in keys
        assert "year" in keys
        assert len(keys) == 4

    def test_values(self):
        """Test values() method."""
        doc = Document(id="doc1", score=0.9, title="Test")
        values = doc.values()
        assert "doc1" in values
        assert 0.9 in values
        assert "Test" in values

    def test_items(self):
        """Test items() method."""
        doc = Document(id="doc1", score=0.9, title="Test")
        items = doc.items()
        assert ("id", "doc1") in items
        assert ("score", 0.9) in items
        assert ("title", "Test") in items

    def test_contains_standard_fields(self):
        """Test __contains__ for standard fields."""
        doc = Document(id="doc1", score=0.9)
        assert "id" in doc
        assert "score" in doc

    def test_contains_dynamic_fields(self):
        """Test __contains__ for dynamic fields."""
        doc = Document(id="doc1", score=0.9, title="Test")
        assert "title" in doc
        assert "nonexistent" not in doc

    def test_iter(self):
        """Test iteration over field names."""
        doc = Document(id="doc1", score=0.9, title="Test", year=2024)
        field_names = list(doc)
        assert "id" in field_names
        assert "score" in field_names
        assert "title" in field_names
        assert "year" in field_names

    def test_len(self):
        """Test __len__ returns correct count."""
        doc = Document(id="doc1", score=0.9)
        assert len(doc) == 2

        doc_with_fields = Document(id="doc1", score=0.9, title="Test", year=2024)
        assert len(doc_with_fields) == 4

    def test_repr_minimal(self):
        """Test __repr__ with only standard fields."""
        doc = Document(id="doc1", score=0.9)
        repr_str = repr(doc)
        assert "Document" in repr_str
        assert "doc1" in repr_str
        assert "0.9" in repr_str

    def test_repr_with_fields(self):
        """Test __repr__ with dynamic fields."""
        doc = Document(id="doc1", score=0.9, title="Test")
        repr_str = repr(doc)
        assert "Document" in repr_str
        assert "doc1" in repr_str
        assert "title" in repr_str
        assert "Test" in repr_str

    def test_equality(self):
        """Test __eq__ between documents."""
        doc1 = Document(id="doc1", score=0.9, title="Test")
        doc2 = Document(id="doc1", score=0.9, title="Test")
        doc3 = Document(id="doc2", score=0.9, title="Test")

        assert doc1 == doc2
        assert doc1 != doc3

    def test_equality_different_types(self):
        """Test __eq__ with non-Document types."""
        doc = Document(id="doc1", score=0.9)
        assert doc != "not a document"
        assert doc != {"id": "doc1", "score": 0.9}

    def test_to_dict(self):
        """Test to_dict() conversion."""
        doc = Document(id="doc1", score=0.9, title="Test", year=2024)
        d = doc.to_dict()
        assert d == {"id": "doc1", "score": 0.9, "title": "Test", "year": 2024}

    def test_to_dict_minimal(self):
        """Test to_dict() with only standard fields."""
        doc = Document(id="doc1", score=0.9)
        d = doc.to_dict()
        assert d == {"id": "doc1", "score": 0.9}


class TestDocumentSearchResponse:
    """Tests for the DocumentSearchResponse class."""

    def test_basic_construction(self):
        """Test creating a response with documents."""
        docs = [
            Document(id="doc1", score=0.95, title="First"),
            Document(id="doc2", score=0.85, title="Second"),
        ]
        response = DocumentSearchResponse(documents=docs)
        assert len(response.documents) == 2
        assert response.documents[0].id == "doc1"
        assert response.documents[1].id == "doc2"

    def test_empty_documents(self):
        """Test response with empty document list."""
        response = DocumentSearchResponse(documents=[])
        assert len(response.documents) == 0

    def test_usage_none_by_default(self):
        """Test that usage is None by default."""
        response = DocumentSearchResponse(documents=[])
        assert response.usage is None

    def test_dict_like_access(self):
        """Test dict-style access to fields."""
        docs = [Document(id="doc1", score=0.9)]
        response = DocumentSearchResponse(documents=docs)
        assert response["documents"] == docs
        assert response.get("usage") is None

    def test_dict_like_get_with_default(self):
        """Test get() with default value."""
        response = DocumentSearchResponse(documents=[])
        assert response.get("nonexistent", "default") == "default"


class TestDocumentUsageExamples:
    """Test the usage examples from the ticket."""

    def test_ticket_example(self):
        """Test the example from the ticket description."""
        # Simulate what results from index.search_documents() would look like
        doc = Document(id="movie-123", score=0.95, title="The Pink Panther", year=1963)

        # Standard field access
        assert doc.id == "movie-123"
        assert doc.score == 0.95

        # Attribute access to dynamic field
        assert doc.title == "The Pink Panther"

        # Dict access to dynamic field
        assert doc["title"] == "The Pink Panther"

        # Safe access with default
        assert doc.get("title", "N/A") == "The Pink Panther"
        assert doc.get("director", "Unknown") == "Unknown"

    def test_iterate_over_results(self):
        """Test iterating over search results."""
        docs = [
            Document(id="doc1", score=0.95, title="First"),
            Document(id="doc2", score=0.85, title="Second"),
            Document(id="doc3", score=0.75, title="Third"),
        ]
        response = DocumentSearchResponse(documents=docs)

        ids = [doc.id for doc in response.documents]
        assert ids == ["doc1", "doc2", "doc3"]

        scores = [doc.score for doc in response.documents]
        assert scores == [0.95, 0.85, 0.75]
