"""Integration tests for backwards compatibility.

These tests verify that existing query(), search(), and upsert() methods
continue to work with schema-based indexes.
"""

import pytest
import os
import uuid
from pinecone import Vector
from tests.integration.helpers import embedding_values, poll_until_lsn_reconciled

FTS_INDEX_DIMENSION = 8


@pytest.mark.skipif(os.getenv("USE_GRPC") != "false", reason="These tests are for REST client only")
class TestQueryMethodCompatibility:
    """Test that existing query() method still works with FTS indexes."""

    def test_query_with_vector(self, fts_index, seeded_fts_namespace):
        """Test that query() works with a vector on FTS index."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        response = fts_index.query(
            namespace=seeded_fts_namespace, vector=query_vector, top_k=5, include_metadata=True
        )

        assert response is not None
        assert hasattr(response, "matches")
        assert len(response.matches) >= 1

        for match in response.matches:
            assert match.id is not None
            assert match.score is not None

    def test_query_with_filter(self, fts_index, seeded_fts_namespace):
        """Test that query() with filter works on FTS index."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        response = fts_index.query(
            namespace=seeded_fts_namespace,
            vector=query_vector,
            top_k=5,
            filter={"category": {"$eq": "comedy"}},
            include_metadata=True,
        )

        assert response is not None
        for match in response.matches:
            if match.metadata:
                assert match.metadata.get("category") == "comedy"

    def test_query_with_include_values(self, fts_index, seeded_fts_namespace):
        """Test that query() with include_values works on FTS index."""
        query_vector = embedding_values(FTS_INDEX_DIMENSION)

        response = fts_index.query(
            namespace=seeded_fts_namespace, vector=query_vector, top_k=3, include_values=True
        )

        assert response is not None
        assert len(response.matches) >= 1


@pytest.mark.skipif(os.getenv("USE_GRPC") != "false", reason="These tests are for REST client only")
class TestUpsertMethodCompatibility:
    """Test that existing upsert() method still works with FTS indexes."""

    def test_upsert_with_vector_object(self, fts_index):
        """Test that upsert() works with Vector objects on FTS index."""
        namespace = f"compat-upsert-vec-{str(uuid.uuid4())[:8]}"

        vectors = [
            Vector(
                id="compat-vec-1",
                values=embedding_values(FTS_INDEX_DIMENSION),
                metadata={"title": "Test Vector 1", "category": "test", "year": 2025},
            ),
            Vector(
                id="compat-vec-2",
                values=embedding_values(FTS_INDEX_DIMENSION),
                metadata={"title": "Test Vector 2", "category": "test", "year": 2025},
            ),
        ]

        response = fts_index.upsert(namespace=namespace, vectors=vectors)

        assert response.upserted_count == 2

    def test_upsert_with_tuple(self, fts_index):
        """Test that upsert() works with tuple format on FTS index."""
        namespace = f"compat-upsert-tuple-{str(uuid.uuid4())[:8]}"

        vectors = [
            ("tuple-vec-1", embedding_values(FTS_INDEX_DIMENSION)),
            ("tuple-vec-2", embedding_values(FTS_INDEX_DIMENSION)),
        ]

        response = fts_index.upsert(namespace=namespace, vectors=vectors)

        assert response.upserted_count == 2

    def test_upsert_with_tuple_and_metadata(self, fts_index):
        """Test that upsert() works with tuple format including metadata."""
        namespace = f"compat-upsert-meta-{str(uuid.uuid4())[:8]}"

        vectors = [
            (
                "meta-vec-1",
                embedding_values(FTS_INDEX_DIMENSION),
                {"title": "Metadata Test", "category": "compat", "year": 2024},
            )
        ]

        response = fts_index.upsert(namespace=namespace, vectors=vectors)

        assert response.upserted_count == 1

    def test_upsert_with_dict(self, fts_index):
        """Test that upsert() works with dict format on FTS index."""
        namespace = f"compat-upsert-dict-{str(uuid.uuid4())[:8]}"

        vectors = [
            {
                "id": "dict-vec-1",
                "values": embedding_values(FTS_INDEX_DIMENSION),
                "metadata": {"title": "Dict Vector", "category": "dict", "year": 2023},
            }
        ]

        response = fts_index.upsert(namespace=namespace, vectors=vectors)

        assert response.upserted_count == 1

    def test_upsert_and_query_roundtrip(self, fts_index):
        """Test that upserted vectors can be queried."""
        namespace = f"compat-roundtrip-{str(uuid.uuid4())[:8]}"
        test_values = embedding_values(FTS_INDEX_DIMENSION)

        vectors = [
            Vector(
                id="roundtrip-vec",
                values=test_values,
                metadata={"title": "Roundtrip Test", "category": "roundtrip", "year": 2025},
            )
        ]

        upsert_response = fts_index.upsert(namespace=namespace, vectors=vectors)
        assert upsert_response.upserted_count == 1

        poll_until_lsn_reconciled(fts_index, upsert_response._response_info, namespace=namespace)

        query_response = fts_index.query(
            namespace=namespace, vector=test_values, top_k=1, include_metadata=True
        )

        assert len(query_response.matches) == 1
        assert query_response.matches[0].id == "roundtrip-vec"


@pytest.mark.skipif(os.getenv("USE_GRPC") != "false", reason="These tests are for REST client only")
class TestFetchMethodCompatibility:
    """Test that existing fetch() method still works with FTS indexes."""

    def test_fetch_by_id(self, fts_index, seeded_fts_namespace):
        """Test that fetch() works on FTS index."""
        response = fts_index.fetch(namespace=seeded_fts_namespace, ids=["movie-1", "movie-2"])

        assert response is not None
        assert hasattr(response, "vectors")
        assert len(response.vectors) >= 1


@pytest.mark.skipif(os.getenv("USE_GRPC") != "false", reason="These tests are for REST client only")
class TestDeleteMethodCompatibility:
    """Test that existing delete() method still works with FTS indexes."""

    def test_delete_by_id(self, fts_index):
        """Test that delete() by id works on FTS index."""
        namespace = f"compat-delete-{str(uuid.uuid4())[:8]}"

        vectors = [
            Vector(
                id="delete-me",
                values=embedding_values(FTS_INDEX_DIMENSION),
                metadata={"title": "To Delete", "category": "delete", "year": 2025},
            )
        ]

        upsert_response = fts_index.upsert(namespace=namespace, vectors=vectors)
        poll_until_lsn_reconciled(fts_index, upsert_response._response_info, namespace=namespace)

        delete_response = fts_index.delete(namespace=namespace, ids=["delete-me"])

        assert delete_response == {}


@pytest.mark.skipif(os.getenv("USE_GRPC") != "false", reason="These tests are for REST client only")
class TestDescribeIndexStatsCompatibility:
    """Test that describe_index_stats() still works with FTS indexes."""

    def test_describe_index_stats(self, fts_index):
        """Test that describe_index_stats() returns expected structure."""
        stats = fts_index.describe_index_stats()

        assert stats is not None
        assert hasattr(stats, "dimension")
        assert hasattr(stats, "total_vector_count")
        assert hasattr(stats, "namespaces")
        assert stats.dimension == FTS_INDEX_DIMENSION
