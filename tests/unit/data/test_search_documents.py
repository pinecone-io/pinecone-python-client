"""Tests for search_documents functionality."""

from unittest.mock import MagicMock

from pinecone.db_data.request_factory import IndexRequestFactory
from pinecone.db_data.dataclasses import (
    TextQuery,
    VectorQuery,
    DocumentSearchResponse,
    Document,
    SparseValues,
)
from pinecone.core.openapi.db_data.models import DocumentSearchRequest


class TestSearchDocumentsRequestFactory:
    """Tests for the search_documents_request factory method."""

    def test_text_query_basic(self):
        """Test request creation with basic text query."""
        request = IndexRequestFactory.search_documents_request(
            score_by=TextQuery(field="title", query="pink panther"), top_k=10
        )
        assert isinstance(request, DocumentSearchRequest)
        assert request.top_k == 10
        assert request.score_by is not None
        assert len(request.score_by) == 1
        assert request.score_by[0]["type"] == "text"
        assert request.score_by[0]["field"] == "title"
        assert request.score_by[0]["text_query"] == "pink panther"

    def test_text_query_with_boost_and_slop(self):
        """Test request creation with text query boost and slop."""
        request = IndexRequestFactory.search_documents_request(
            score_by=TextQuery(field="title", query="pink panther", boost=2.0, slop=3), top_k=5
        )
        assert request.top_k == 5
        assert request.score_by[0]["boost"] == 2.0
        assert request.score_by[0]["slop"] == 3

    def test_vector_query_dense(self):
        """Test request creation with dense vector query."""
        request = IndexRequestFactory.search_documents_request(
            score_by=VectorQuery(field="embedding", values=[0.1, 0.2, 0.3]), top_k=20
        )
        assert isinstance(request, DocumentSearchRequest)
        assert request.top_k == 20
        assert request.score_by is not None
        assert len(request.score_by) == 1
        assert request.score_by[0]["type"] == "vector"
        assert request.score_by[0]["field"] == "embedding"
        assert request.score_by[0]["values"] == [0.1, 0.2, 0.3]

    def test_vector_query_sparse(self):
        """Test request creation with sparse vector query."""
        sparse = SparseValues(indices=[1, 5, 10], values=[0.5, 0.3, 0.2])
        request = IndexRequestFactory.search_documents_request(
            score_by=VectorQuery(field="sparse_embedding", sparse_values=sparse), top_k=10
        )
        assert request.score_by[0]["type"] == "vector"
        assert request.score_by[0]["field"] == "sparse_embedding"
        assert "sparse_values" in request.score_by[0]

    def test_with_filter(self):
        """Test request creation with metadata filter."""
        request = IndexRequestFactory.search_documents_request(
            score_by=TextQuery(field="title", query="panther"),
            top_k=10,
            filter={"genre": {"$eq": "comedy"}},
        )
        assert request.filter == {"genre": {"$eq": "comedy"}}

    def test_with_include_fields_wildcard(self):
        """Test request creation with wildcard include_fields."""
        request = IndexRequestFactory.search_documents_request(
            score_by=TextQuery(field="title", query="panther"), top_k=10, include_fields=["*"]
        )
        assert request.include_fields == "*"

    def test_with_include_fields_list(self):
        """Test request creation with specific include_fields."""
        request = IndexRequestFactory.search_documents_request(
            score_by=TextQuery(field="title", query="panther"),
            top_k=10,
            include_fields=["title", "year", "genre"],
        )
        assert request.include_fields == ["title", "year", "genre"]

    def test_default_top_k(self):
        """Test that default top_k is 10."""
        request = IndexRequestFactory.search_documents_request(
            score_by=TextQuery(field="title", query="panther")
        )
        assert request.top_k == 10

    def test_with_all_parameters(self):
        """Test request creation with all parameters."""
        request = IndexRequestFactory.search_documents_request(
            score_by=TextQuery(field="title", query="panther", boost=1.5),
            top_k=25,
            filter={"year": {"$gte": 2000}},
            include_fields=["title", "year"],
        )
        assert request.top_k == 25
        assert request.filter == {"year": {"$gte": 2000}}
        assert request.include_fields == ["title", "year"]
        assert request.score_by[0]["boost"] == 1.5


class TestDocumentClass:
    """Tests for the Document class."""

    def test_document_creation(self):
        """Test creating a Document with id, score, and fields."""
        doc = Document(id="doc1", score=0.95, title="Test Title", year=2020)
        assert doc.id == "doc1"
        assert doc.score == 0.95
        assert doc.title == "Test Title"
        assert doc.year == 2020

    def test_document_dict_access(self):
        """Test dict-style access to Document fields."""
        doc = Document(id="doc1", score=0.95, title="Test Title")
        assert doc["id"] == "doc1"
        assert doc["score"] == 0.95
        assert doc["title"] == "Test Title"

    def test_document_get_with_default(self):
        """Test get() method with default value."""
        doc = Document(id="doc1", score=0.95)
        assert doc.get("title") is None
        assert doc.get("title", "N/A") == "N/A"
        assert doc.get("id") == "doc1"

    def test_document_contains(self):
        """Test 'in' operator for Document."""
        doc = Document(id="doc1", score=0.95, title="Test")
        assert "id" in doc
        assert "score" in doc
        assert "title" in doc
        assert "missing" not in doc

    def test_document_keys(self):
        """Test keys() method."""
        doc = Document(id="doc1", score=0.95, title="Test")
        assert doc.keys() == ["id", "score", "title"]

    def test_document_to_dict(self):
        """Test to_dict() method."""
        doc = Document(id="doc1", score=0.95, title="Test", year=2020)
        result = doc.to_dict()
        assert result == {"id": "doc1", "score": 0.95, "title": "Test", "year": 2020}


class TestDocumentSearchResponse:
    """Tests for the DocumentSearchResponse class."""

    def test_response_creation(self):
        """Test creating a DocumentSearchResponse."""
        docs = [
            Document(id="doc1", score=0.95, title="Title 1"),
            Document(id="doc2", score=0.85, title="Title 2"),
        ]
        response = DocumentSearchResponse(documents=docs)
        assert len(response.documents) == 2
        assert response.documents[0].id == "doc1"
        assert response.documents[1].id == "doc2"

    def test_response_with_usage(self):
        """Test DocumentSearchResponse with usage info."""
        docs = [Document(id="doc1", score=0.95)]
        # Mock usage object
        usage = MagicMock()
        usage.read_units = 5
        response = DocumentSearchResponse(documents=docs, usage=usage)
        assert response.usage.read_units == 5

    def test_empty_response(self):
        """Test DocumentSearchResponse with no documents."""
        response = DocumentSearchResponse(documents=[])
        assert len(response.documents) == 0
