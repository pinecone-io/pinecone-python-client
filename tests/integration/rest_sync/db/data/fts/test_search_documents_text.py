"""Integration tests for search_documents() with text queries.

These tests verify full-text search functionality using TextQuery.
"""

import pytest
import os
from pinecone import text_query, TextQuery


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestSearchDocumentsWithTextQuery:
    """Test search_documents() with various text query patterns."""

    def test_simple_text_search(self, fts_index, seeded_fts_namespace):
        """Test simple text search returns matching documents."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "pink panther"), top_k=10
        )

        assert len(results.documents) >= 1
        assert results.usage is not None

        # All results should contain "pink" or "panther" in the title
        for doc in results.documents:
            assert doc.id is not None
            assert doc.score >= 0
            title_lower = doc.title.lower()
            assert "pink" in title_lower or "panther" in title_lower

    def test_text_search_with_phrase_matching(self, fts_index, seeded_fts_namespace):
        """Test phrase matching with quoted strings."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", '"Pink Panther"'), top_k=10
        )

        assert len(results.documents) >= 1
        # Phrase match should find "Pink Panther" as an exact phrase
        for doc in results.documents:
            assert "Pink Panther" in doc.title

    def test_text_search_with_required_terms(self, fts_index, seeded_fts_namespace):
        """Test required terms with +term syntax."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace, score_by=text_query("title", "+Return +Pink"), top_k=10
        )

        # Should find "Return of the Pink Panther"
        assert len(results.documents) >= 1
        for doc in results.documents:
            title_lower = doc.title.lower()
            assert "return" in title_lower
            assert "pink" in title_lower

    def test_text_search_with_boost(self, fts_index, seeded_fts_namespace):
        """Test boost parameter for relevance scoring."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther", boost=2.0),
            top_k=10,
        )

        assert len(results.documents) >= 1
        # Verify boost doesn't break the query
        for doc in results.documents:
            assert doc.score >= 0

    def test_text_search_with_slop(self, fts_index, seeded_fts_namespace):
        """Test slop parameter for phrase proximity."""
        # With slop=1, "Return Panther" should match "Return of the Pink Panther"
        # because terms can be 1 position apart
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", '"Pink Panther"', slop=2),
            top_k=10,
        )

        assert len(results.documents) >= 1

    def test_text_search_using_class_directly(self, fts_index, seeded_fts_namespace):
        """Test using TextQuery class directly instead of helper function."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=TextQuery(field="title", query="Matrix"),
            top_k=10,
        )

        assert len(results.documents) >= 1
        for doc in results.documents:
            assert "Matrix" in doc.title

    def test_text_search_on_description_field(self, fts_index, seeded_fts_namespace):
        """Test text search on a different full-text searchable field."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("description", "investigates"),
            top_k=10,
        )

        assert len(results.documents) >= 1
        for doc in results.documents:
            assert "investigates" in doc.description.lower()


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestSearchDocumentsWithTextFiltering:
    """Test search_documents() with text queries and metadata filtering."""

    def test_text_search_with_category_filter(self, fts_index, seeded_fts_namespace):
        """Test text search combined with category filter."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            filter={"category": {"$eq": "comedy"}},
            top_k=10,
        )

        assert len(results.documents) >= 1
        for doc in results.documents:
            assert doc.category == "comedy"

    def test_text_search_with_year_filter(self, fts_index, seeded_fts_namespace):
        """Test text search combined with year filter."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            filter={"year": {"$gte": 1976}},
            top_k=10,
        )

        assert len(results.documents) >= 1
        for doc in results.documents:
            assert doc.year >= 1976

    def test_text_search_with_combined_filters(self, fts_index, seeded_fts_namespace):
        """Test text search with multiple filter conditions."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            filter={"category": {"$eq": "comedy"}, "year": {"$lte": 1976}},
            top_k=10,
        )

        assert len(results.documents) >= 1
        for doc in results.documents:
            assert doc.category == "comedy"
            assert doc.year <= 1976

    def test_text_search_with_text_match_filter(self, fts_index, seeded_fts_namespace):
        """Test using $text_match operator in filter."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            filter={"description": {"$text_match": "investigates"}},
            top_k=10,
        )

        assert len(results.documents) >= 1
        for doc in results.documents:
            assert "investigates" in doc.description.lower()


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestSearchDocumentsWithIncludeFields:
    """Test search_documents() with include_fields parameter."""

    def test_include_specific_fields(self, fts_index, seeded_fts_namespace):
        """Test including only specific fields in response."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["title", "year"],
            top_k=10,
        )

        assert len(results.documents) >= 1
        for doc in results.documents:
            assert "title" in doc
            assert "year" in doc
            # Other fields should not be present
            assert "description" not in doc or doc.get("description") is None
            assert "category" not in doc or doc.get("category") is None

    def test_include_all_fields(self, fts_index, seeded_fts_namespace):
        """Test including all fields with wildcard."""
        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=text_query("title", "panther"),
            include_fields=["*"],
            top_k=10,
        )

        assert len(results.documents) >= 1
        for doc in results.documents:
            assert "title" in doc
            assert "description" in doc
            assert "category" in doc
            assert "year" in doc
