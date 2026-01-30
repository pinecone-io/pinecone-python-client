"""Tests for query factory functions."""

from pinecone.db_data.query_helpers import text_query, vector_query
from pinecone.db_data.dataclasses import TextQuery, VectorQuery, SparseValues


class TestTextQueryFactory:
    """Tests for the text_query factory function."""

    def test_required_params(self):
        """Factory creates TextQuery with required parameters."""
        result = text_query(field="title", query="pink panther")
        assert isinstance(result, TextQuery)
        assert result.field == "title"
        assert result.query == "pink panther"
        assert result.boost is None
        assert result.slop is None

    def test_with_boost(self):
        """Factory creates TextQuery with boost parameter."""
        result = text_query(field="title", query="pink panther", boost=2.5)
        assert isinstance(result, TextQuery)
        assert result.field == "title"
        assert result.query == "pink panther"
        assert result.boost == 2.5
        assert result.slop is None

    def test_with_slop(self):
        """Factory creates TextQuery with slop parameter."""
        result = text_query(field="title", query="pink panther", slop=3)
        assert isinstance(result, TextQuery)
        assert result.field == "title"
        assert result.query == "pink panther"
        assert result.boost is None
        assert result.slop == 3

    def test_with_all_options(self):
        """Factory creates TextQuery with all optional parameters."""
        result = text_query(field="title", query="pink panther", boost=1.5, slop=2)
        assert isinstance(result, TextQuery)
        assert result.field == "title"
        assert result.query == "pink panther"
        assert result.boost == 1.5
        assert result.slop == 2

    def test_phrase_query(self):
        """Factory handles phrase queries with quotes."""
        result = text_query(field="title", query='"pink panther"')
        assert result.query == '"pink panther"'

    def test_required_terms_query(self):
        """Factory handles required term syntax."""
        result = text_query(field="title", query="+return +panther")
        assert result.query == "+return +panther"

    def test_as_dict_output(self):
        """Factory result produces correct as_dict output."""
        result = text_query(field="title", query="test", boost=2.0, slop=1)
        as_dict = result.as_dict()
        assert as_dict == {
            "type": "text",
            "field": "title",
            "text_query": "test",
            "boost": 2.0,
            "slop": 1,
        }


class TestVectorQueryFactory:
    """Tests for the vector_query factory function."""

    def test_required_params(self):
        """Factory creates VectorQuery with required parameters only."""
        result = vector_query(field="embedding")
        assert isinstance(result, VectorQuery)
        assert result.field == "embedding"
        assert result.values is None
        assert result.sparse_values is None

    def test_with_dense_values(self):
        """Factory creates VectorQuery with dense values."""
        result = vector_query(field="embedding", values=[0.1, 0.2, 0.3])
        assert isinstance(result, VectorQuery)
        assert result.field == "embedding"
        assert result.values == [0.1, 0.2, 0.3]
        assert result.sparse_values is None

    def test_with_sparse_values(self):
        """Factory creates VectorQuery with sparse values."""
        sparse = SparseValues(indices=[1, 5, 10], values=[0.5, 0.3, 0.2])
        result = vector_query(field="sparse_embedding", sparse_values=sparse)
        assert isinstance(result, VectorQuery)
        assert result.field == "sparse_embedding"
        assert result.values is None
        assert result.sparse_values is sparse

    def test_with_both_values(self):
        """Factory creates VectorQuery with both dense and sparse values."""
        sparse = SparseValues(indices=[1, 2], values=[0.5, 0.5])
        result = vector_query(field="hybrid", values=[0.1, 0.2, 0.3], sparse_values=sparse)
        assert isinstance(result, VectorQuery)
        assert result.field == "hybrid"
        assert result.values == [0.1, 0.2, 0.3]
        assert result.sparse_values is sparse

    def test_as_dict_output_dense(self):
        """Factory result produces correct as_dict output for dense vectors."""
        result = vector_query(field="embedding", values=[0.1, 0.2, 0.3])
        as_dict = result.as_dict()
        assert as_dict == {"type": "vector", "field": "embedding", "values": [0.1, 0.2, 0.3]}

    def test_as_dict_output_sparse(self):
        """Factory result produces correct as_dict output for sparse vectors."""
        sparse = SparseValues(indices=[1, 5, 10], values=[0.5, 0.3, 0.2])
        result = vector_query(field="sparse", sparse_values=sparse)
        as_dict = result.as_dict()
        assert as_dict == {
            "type": "vector",
            "field": "sparse",
            "sparse_values": {"indices": [1, 5, 10], "values": [0.5, 0.3, 0.2]},
        }


class TestFactoryTopLevelExports:
    """Tests that factory functions are exported from pinecone package."""

    def test_text_query_importable(self):
        """text_query is importable from pinecone package."""
        from pinecone import text_query as imported_text_query

        result = imported_text_query(field="test", query="query")
        assert isinstance(result, TextQuery)

    def test_vector_query_importable(self):
        """vector_query is importable from pinecone package."""
        from pinecone import vector_query as imported_vector_query

        result = imported_vector_query(field="test")
        assert isinstance(result, VectorQuery)
