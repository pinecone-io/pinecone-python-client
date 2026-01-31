"""Integration tests for search_documents() with vector queries.

These tests verify vector similarity search functionality using VectorQuery.
"""

import pytest
import os
from pinecone import vector_query, VectorQuery
from tests.integration.helpers import embedding_values

FTS_INDEX_DIMENSION = 8


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestSearchDocumentsWithVectorQuery:
    """Test search_documents() with dense vector queries."""

    def test_dense_vector_search(self, fts_index, seeded_fts_namespace):
        """Test dense vector similarity search."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=vector_query("embedding", values=query_vector),
            top_k=10,
        )

        assert len(results.documents) >= 1
        assert results.usage is not None

        for doc in results.documents:
            assert doc.id is not None
            assert doc.score >= 0

    def test_vector_search_using_class_directly(self, fts_index, seeded_fts_namespace):
        """Test using VectorQuery class directly instead of helper function."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=VectorQuery(field="embedding", values=query_vector),
            top_k=5,
        )

        assert len(results.documents) >= 1
        assert len(results.documents) <= 5

    def test_vector_search_with_top_k(self, fts_index, seeded_fts_namespace):
        """Test that top_k limits the number of results."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=vector_query("embedding", values=query_vector),
            top_k=2,
        )

        assert len(results.documents) <= 2


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestVectorSearchWithFilters:
    """Test search_documents() with vector queries and metadata filtering."""

    def test_vector_search_with_category_filter(self, fts_index, seeded_fts_namespace):
        """Test vector search combined with category filter."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=vector_query("embedding", values=query_vector),
            filter={"category": {"$eq": "comedy"}},
            top_k=10,
        )

        for doc in results.documents:
            assert doc.category == "comedy"

    def test_vector_search_with_year_filter(self, fts_index, seeded_fts_namespace):
        """Test vector search combined with year range filter."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=vector_query("embedding", values=query_vector),
            filter={"year": {"$gte": 1980}},
            top_k=10,
        )

        for doc in results.documents:
            assert doc.year >= 1980

    def test_vector_search_with_text_match_filter(self, fts_index, seeded_fts_namespace):
        """Test vector search with $text_match filter on text field."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=vector_query("embedding", values=query_vector),
            filter={"title": {"$text_match": "panther"}},
            top_k=10,
        )

        for doc in results.documents:
            assert "panther" in doc.title.lower()

    def test_vector_search_with_combined_filters(self, fts_index, seeded_fts_namespace):
        """Test vector search with multiple filter conditions."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=vector_query("embedding", values=query_vector),
            filter={"category": {"$eq": "scifi"}, "year": {"$gte": 1990}},
            top_k=10,
        )

        for doc in results.documents:
            assert doc.category == "scifi"
            assert doc.year >= 1990


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="Document operations not supported in gRPC"
)
class TestVectorSearchWithIncludeFields:
    """Test search_documents() with vector queries and include_fields."""

    def test_vector_search_include_specific_fields(self, fts_index, seeded_fts_namespace):
        """Test vector search with specific fields included."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=vector_query("embedding", values=query_vector),
            include_fields=["title", "category"],
            top_k=10,
        )

        assert len(results.documents) >= 1
        for doc in results.documents:
            assert "title" in doc
            assert "category" in doc

    def test_vector_search_include_all_fields(self, fts_index, seeded_fts_namespace):
        """Test vector search with all fields included."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        results = fts_index.search_documents(
            namespace=seeded_fts_namespace,
            score_by=vector_query("embedding", values=query_vector),
            include_fields=["*"],
            top_k=10,
        )

        assert len(results.documents) >= 1
        for doc in results.documents:
            assert "title" in doc
            assert "description" in doc
            assert "category" in doc
            assert "year" in doc
