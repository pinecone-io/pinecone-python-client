import pytest
from pinecone.db_data.request_factory import (
    IndexRequestFactory,
    SearchQuery,
    SearchQueryVector,
    SearchRerank,
)

from pinecone.core.openapi.db_data.models import (
    SearchRecordsRequestQuery,
    SearchRecordsRequestRerank,
    SearchRecordsVector,
    VectorValues,
    SearchRecordsRequest,
)

from pinecone import RerankModel


class TestIndexRequestFactory:
    def test_parsing_search_query_vector_dict(self):
        vector1 = IndexRequestFactory._parse_search_vector(None)
        assert vector1 is None

        vector2 = IndexRequestFactory._parse_search_vector({"values": [0.1, 0.2, 0.3]})
        assert vector2 == SearchRecordsVector(values=VectorValues([0.1, 0.2, 0.3]))

        vector3 = IndexRequestFactory._parse_search_vector(
            {
                "values": [0.1, 0.2, 0.3],
                "sparse_values": [0.4, 0.5, 0.6],
                "sparse_indices": [1, 2, 3],
            }
        )
        assert vector3 == SearchRecordsVector(
            values=VectorValues([0.1, 0.2, 0.3]),
            sparse_indices=[1, 2, 3],
            sparse_values=[0.4, 0.5, 0.6],
        )

        vector4 = IndexRequestFactory._parse_search_vector(
            {"values": [], "sparse_values": [0.4, 0.5, 0.6], "sparse_indices": [1, 2, 3]}
        )
        assert vector4 == SearchRecordsVector(
            values=VectorValues([]), sparse_indices=[1, 2, 3], sparse_values=[0.4, 0.5, 0.6]
        )

        vector5 = IndexRequestFactory._parse_search_vector(
            {"values": [0.1, 0.2, 0.3], "sparse_values": [], "sparse_indices": []}
        )
        assert vector5 == SearchRecordsVector(
            values=VectorValues([0.1, 0.2, 0.3]), sparse_indices=[], sparse_values=[]
        )

        vector6 = IndexRequestFactory._parse_search_vector(
            {"values": [], "sparse_values": [], "sparse_indices": []}
        )
        assert vector6 == SearchRecordsVector(
            values=VectorValues([]), sparse_indices=[], sparse_values=[]
        )

        vector7 = IndexRequestFactory._parse_search_vector({})
        assert vector7 is None

        vector8 = IndexRequestFactory._parse_search_vector({"values": []})
        assert vector8 == SearchRecordsVector(values=VectorValues([]))

        vector9 = IndexRequestFactory._parse_search_vector(
            {"sparse_values": [0.8], "sparse_indices": [100]}
        )
        assert vector9 == SearchRecordsVector(sparse_indices=[100], sparse_values=[0.8])

    def test_parsing_search_query_vector_object(self):
        vector2 = IndexRequestFactory._parse_search_vector(
            SearchQueryVector(values=[0.1, 0.2, 0.3])
        )
        assert vector2 == SearchRecordsVector(values=VectorValues([0.1, 0.2, 0.3]))

        vector3 = IndexRequestFactory._parse_search_vector(
            SearchQueryVector(
                values=[0.1, 0.2, 0.3], sparse_values=[0.4, 0.5, 0.6], sparse_indices=[1, 2, 3]
            )
        )
        assert vector3 == SearchRecordsVector(
            values=VectorValues([0.1, 0.2, 0.3]),
            sparse_indices=[1, 2, 3],
            sparse_values=[0.4, 0.5, 0.6],
        )

        vector4 = IndexRequestFactory._parse_search_vector(
            SearchQueryVector(values=[], sparse_values=[0.4, 0.5, 0.6], sparse_indices=[1, 2, 3])
        )
        assert vector4 == SearchRecordsVector(
            values=VectorValues([]), sparse_indices=[1, 2, 3], sparse_values=[0.4, 0.5, 0.6]
        )

        vector5 = IndexRequestFactory._parse_search_vector(
            SearchQueryVector(values=[0.1, 0.2, 0.3], sparse_values=[], sparse_indices=[])
        )
        assert vector5 == SearchRecordsVector(
            values=VectorValues([0.1, 0.2, 0.3]), sparse_indices=[], sparse_values=[]
        )

        vector6 = IndexRequestFactory._parse_search_vector(
            SearchQueryVector(values=[], sparse_values=[], sparse_indices=[])
        )
        assert vector6 == SearchRecordsVector(
            values=VectorValues([]), sparse_indices=[], sparse_values=[]
        )

        vector7 = IndexRequestFactory._parse_search_vector(SearchQueryVector(values=[]))
        assert vector7 == SearchRecordsVector(values=VectorValues([]))

    def test_parse_search_query_dict(self):
        query1 = IndexRequestFactory._parse_search_query(
            {
                "inputs": {"text": "Apple corporation"},
                "top_k": 3,
                "vector": {"values": [0.1, 0.2, 0.3]},
            }
        )
        assert query1 == SearchRecordsRequestQuery(
            inputs={"text": "Apple corporation"},
            top_k=3,
            vector=SearchRecordsVector(values=VectorValues([0.1, 0.2, 0.3])),
        )

        query2 = IndexRequestFactory._parse_search_query(
            {
                "inputs": {"text": "Apple corporation"},
                "top_k": 3,
                "vector": {
                    "values": [0.1, 0.2, 0.3],
                    "sparse_values": [0.4, 0.5, 0.6],
                    "sparse_indices": [1, 2, 3],
                },
            }
        )
        assert query2 == SearchRecordsRequestQuery(
            inputs={"text": "Apple corporation"},
            top_k=3,
            vector=SearchRecordsVector(
                values=VectorValues([0.1, 0.2, 0.3]),
                sparse_indices=[1, 2, 3],
                sparse_values=[0.4, 0.5, 0.6],
            ),
        )

        query3 = IndexRequestFactory._parse_search_query(
            {
                "inputs": {"text": "Apple corporation"},
                "top_k": 3,
                "id": "test_id",
                "filter": {"genre": {"$in": ["action"]}},
            }
        )
        assert query3 == SearchRecordsRequestQuery(
            inputs={"text": "Apple corporation"},
            top_k=3,
            id="test_id",
            filter={"genre": {"$in": ["action"]}},
        )

    def test_parse_search_query_missing_required_fields(self):
        with pytest.raises(ValueError) as e:
            IndexRequestFactory._parse_search_query({"inputs": {"text": "Apple corporation"}})
        assert str(e.value) == "Missing required field 'top_k' in search query."

    def test_parse_search_query_with_objects(self):
        query = IndexRequestFactory._parse_search_query(
            SearchQuery(
                inputs={"text": "Apple corporation"},
                top_k=3,
                filter={"genre": {"$in": ["action"]}},
                id="test_id",
                vector=SearchQueryVector(
                    values=[0.1, 0.2, 0.3], sparse_indices=[1, 2, 3], sparse_values=[0.4, 0.5, 0.6]
                ),
            )
        )
        assert query == SearchRecordsRequestQuery(
            inputs={"text": "Apple corporation"},
            top_k=3,
            id="test_id",
            filter={"genre": {"$in": ["action"]}},
            vector=SearchRecordsVector(
                values=VectorValues([0.1, 0.2, 0.3]),
                sparse_indices=[1, 2, 3],
                sparse_values=[0.4, 0.5, 0.6],
            ),
        )

    def test_parse_search_query_with_mixed_types(self):
        query = IndexRequestFactory._parse_search_query(
            SearchQuery(
                inputs={"text": "Apple corporation"},
                top_k=3,
                filter={"genre": {"$in": ["action"]}},
                id="test_id",
                vector={
                    "values": [0.1, 0.2, 0.3],
                    "sparse_indices": [1, 2, 3],
                    "sparse_values": [0.4, 0.5, 0.6],
                },
            )
        )
        assert query == SearchRecordsRequestQuery(
            inputs={"text": "Apple corporation"},
            top_k=3,
            id="test_id",
            filter={"genre": {"$in": ["action"]}},
            vector=SearchRecordsVector(
                values=VectorValues([0.1, 0.2, 0.3]),
                sparse_indices=[1, 2, 3],
                sparse_values=[0.4, 0.5, 0.6],
            ),
        )

    def test_parse_search_rerank_dict(self):
        rerank = IndexRequestFactory._parse_search_rerank(
            {"model": "bge-reranker-v2-m3", "rank_fields": ["my_text_field"], "top_n": 3}
        )
        assert rerank == SearchRecordsRequestRerank(
            model="bge-reranker-v2-m3", rank_fields=["my_text_field"], top_n=3
        )

        # Enum used in dict
        rerank2 = IndexRequestFactory._parse_search_rerank(
            {"model": RerankModel.Bge_Reranker_V2_M3, "rank_fields": ["my_text_field"], "top_n": 3}
        )
        assert rerank2 == SearchRecordsRequestRerank(
            model="bge-reranker-v2-m3", rank_fields=["my_text_field"], top_n=3
        )

        rerank3 = IndexRequestFactory._parse_search_rerank(
            {
                "model": "bge-reranker-v2-m3",
                "rank_fields": ["my_text_field"],
                "top_n": 3,
                "parameters": {"key": "value"},
                "query": "foo",
            }
        )
        assert rerank3 == SearchRecordsRequestRerank(
            model="bge-reranker-v2-m3",
            rank_fields=["my_text_field"],
            top_n=3,
            parameters={"key": "value"},
            query="foo",
        )

    def test_parse_search_rerank_missing_required_fields(self):
        with pytest.raises(ValueError) as e:
            IndexRequestFactory._parse_search_rerank({"rank_fields": ["my_text_field"], "top_n": 3})
        assert str(e.value) == "Missing required field 'model' in rerank."

        with pytest.raises(ValueError) as e:
            IndexRequestFactory._parse_search_rerank(
                {"rank_fields": ["my_text_field"], "model": None}
            )
        assert str(e.value) == "Missing required field 'model' in rerank."

        with pytest.raises(ValueError) as e:
            IndexRequestFactory._parse_search_rerank({"rank_fields": None, "model": "foo"})
        assert str(e.value) == "Missing required field 'rank_fields' in rerank."

        with pytest.raises(ValueError) as e:
            IndexRequestFactory._parse_search_rerank({"model": "bge-reranker-v2-m3", "top_n": 3})
        assert str(e.value) == "Missing required field 'rank_fields' in rerank."

        with pytest.raises(ValueError) as e:
            IndexRequestFactory._parse_search_rerank(
                {"model": RerankModel.Bge_Reranker_V2_M3, "top_n": 3}
            )
        assert str(e.value) == "Missing required field 'rank_fields' in rerank."

    def test_parse_search_rerank_object(self):
        rerank = IndexRequestFactory._parse_search_rerank(
            SearchRerank(
                model=RerankModel.Bge_Reranker_V2_M3, rank_fields=["my_text_field"], top_n=3
            )
        )
        assert rerank == SearchRecordsRequestRerank(
            model="bge-reranker-v2-m3", rank_fields=["my_text_field"], top_n=3
        )

        rerank2 = IndexRequestFactory._parse_search_rerank(
            SearchRerank(
                model=RerankModel.Bge_Reranker_V2_M3,
                rank_fields=["my_text_field"],
                top_n=3,
                parameters={"key": "value"},
                query="foo",
            )
        )
        assert rerank2 == SearchRecordsRequestRerank(
            model="bge-reranker-v2-m3",
            rank_fields=["my_text_field"],
            top_n=3,
            parameters={"key": "value"},
            query="foo",
        )

        rerank3 = IndexRequestFactory._parse_search_rerank(
            SearchRerank(model="unknown-model", rank_fields=["my_text_field"])
        )
        assert rerank3 == SearchRecordsRequestRerank(
            model="unknown-model", rank_fields=["my_text_field"]
        )

    def test_search_request_with_dicts(self):
        request = IndexRequestFactory.search_request(
            query={
                "inputs": {"text": "Apple corporation"},
                "top_k": 3,
                "vector": {"values": [0.1, 0.2, 0.3]},
            },
            fields=["more_stuff"],
            rerank={"model": "bge-reranker-v2-m3", "rank_fields": ["my_text_field"], "top_n": 3},
        )

        assert request is not None
        assert request.query.vector == SearchRecordsVector(values=VectorValues([0.1, 0.2, 0.3]))

    def test_search_request_with_objects_and_enums(self):
        factory = IndexRequestFactory()
        request = factory.search_request(
            query=SearchQuery(
                inputs={"text": "Apple corporation"},
                top_k=3,
                vector=SearchQueryVector(values=[0.1, 0.2, 0.3]),
                filter={"genre": {"$in": ["action"]}},
                id="test_id",
            ),
            fields=["more_stuff"],
            rerank=SearchRerank(
                model=RerankModel.Bge_Reranker_V2_M3,
                rank_fields=["my_text_field"],
                top_n=3,
                parameters={"key": "value"},
                query="foo",
            ),
        )

        assert request == SearchRecordsRequest(
            query=SearchRecordsRequestQuery(
                inputs={"text": "Apple corporation"},
                top_k=3,
                vector=SearchRecordsVector(values=VectorValues([0.1, 0.2, 0.3])),
                filter={"genre": {"$in": ["action"]}},
                id="test_id",
            ),
            fields=["more_stuff"],
            rerank=SearchRecordsRequestRerank(
                model="bge-reranker-v2-m3",
                rank_fields=["my_text_field"],
                top_n=3,
                parameters={"key": "value"},
                query="foo",
            ),
        )

    def test_search_request_with_no_rerank(self):
        factory = IndexRequestFactory()
        request = factory.search_request(
            query=SearchQuery(
                inputs={"text": "Apple corporation"},
                top_k=3,
                vector=SearchQueryVector(values=[0.1, 0.2, 0.3]),
                filter={"genre": {"$in": ["action"]}},
                id="test_id",
            ),
            fields=["more_stuff"],
        )

        assert request == SearchRecordsRequest(
            query=SearchRecordsRequestQuery(
                inputs={"text": "Apple corporation"},
                top_k=3,
                vector=SearchRecordsVector(values=VectorValues([0.1, 0.2, 0.3])),
                filter={"genre": {"$in": ["action"]}},
                id="test_id",
            ),
            fields=["more_stuff"],
        )

        request2 = IndexRequestFactory.search_request(
            query={
                "inputs": {"text": "Apple corporation"},
                "top_k": 3,
                "vector": {"values": [0.1, 0.2, 0.3]},
            }
        )
        assert request2 == SearchRecordsRequest(
            query=SearchRecordsRequestQuery(
                inputs={"text": "Apple corporation"},
                top_k=3,
                vector=SearchRecordsVector(values=VectorValues([0.1, 0.2, 0.3])),
            ),
            fields=["*"],
        )
