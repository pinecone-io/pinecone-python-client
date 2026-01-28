"""Tests for TextQuery and VectorQuery classes."""

from pinecone.db_data.dataclasses import TextQuery, VectorQuery, SparseValues


class TestTextQuery:
    def test_required_params(self):
        query = TextQuery(field="title", query="pink panther")
        assert query.field == "title"
        assert query.query == "pink panther"
        assert query.boost is None
        assert query.slop is None

    def test_to_dict_minimal(self):
        query = TextQuery(field="title", query="pink panther")
        result = query.to_dict()
        assert result == {"field": "title", "query": "pink panther"}

    def test_to_dict_with_boost(self):
        query = TextQuery(field="title", query="pink panther", boost=2.0)
        result = query.to_dict()
        assert result == {"field": "title", "query": "pink panther", "boost": 2.0}

    def test_to_dict_with_slop(self):
        query = TextQuery(field="title", query="pink panther", slop=2)
        result = query.to_dict()
        assert result == {"field": "title", "query": "pink panther", "slop": 2}

    def test_to_dict_with_all_options(self):
        query = TextQuery(field="title", query="pink panther", boost=1.5, slop=3)
        result = query.to_dict()
        assert result == {"field": "title", "query": "pink panther", "boost": 1.5, "slop": 3}

    def test_dict_like_access(self):
        query = TextQuery(field="title", query="pink panther", boost=2.0)
        assert query["field"] == "title"
        assert query["query"] == "pink panther"
        assert query["boost"] == 2.0

    def test_dict_like_get(self):
        query = TextQuery(field="title", query="pink panther")
        assert query.get("field") == "title"
        assert query.get("boost") is None
        assert query.get("nonexistent", "default") == "default"


class TestVectorQuery:
    def test_required_params(self):
        query = VectorQuery(field="embedding")
        assert query.field == "embedding"
        assert query.values is None
        assert query.sparse_values is None

    def test_to_dict_minimal(self):
        query = VectorQuery(field="embedding")
        result = query.to_dict()
        assert result == {"field": "embedding"}

    def test_to_dict_with_values(self):
        query = VectorQuery(field="embedding", values=[0.1, 0.2, 0.3])
        result = query.to_dict()
        assert result == {"field": "embedding", "values": [0.1, 0.2, 0.3]}

    def test_to_dict_with_sparse_values(self):
        sparse = SparseValues(indices=[1, 5, 10], values=[0.5, 0.3, 0.2])
        query = VectorQuery(field="sparse_embedding", sparse_values=sparse)
        result = query.to_dict()
        assert result == {
            "field": "sparse_embedding",
            "sparse_values": {"indices": [1, 5, 10], "values": [0.5, 0.3, 0.2]},
        }

    def test_to_dict_with_both_values(self):
        sparse = SparseValues(indices=[1, 2], values=[0.5, 0.5])
        query = VectorQuery(field="hybrid", values=[0.1, 0.2, 0.3], sparse_values=sparse)
        result = query.to_dict()
        assert result == {
            "field": "hybrid",
            "values": [0.1, 0.2, 0.3],
            "sparse_values": {"indices": [1, 2], "values": [0.5, 0.5]},
        }

    def test_dict_like_access(self):
        query = VectorQuery(field="embedding", values=[0.1, 0.2])
        assert query["field"] == "embedding"
        assert query["values"] == [0.1, 0.2]

    def test_dict_like_get(self):
        query = VectorQuery(field="embedding")
        assert query.get("field") == "embedding"
        assert query.get("values") is None
        assert query.get("nonexistent", "default") == "default"


class TestQueryUsageExamples:
    """Test the usage examples from the ticket."""

    def test_text_query_example(self):
        query = TextQuery(field="title", query='return "pink panther"')
        result = query.to_dict()
        assert result["field"] == "title"
        assert result["query"] == 'return "pink panther"'

    def test_vector_query_example(self):
        query = VectorQuery(field="embedding", values=[0.1, 0.2, 0.3])
        result = query.to_dict()
        assert result["field"] == "embedding"
        assert result["values"] == [0.1, 0.2, 0.3]
