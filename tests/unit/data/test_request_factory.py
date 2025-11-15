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
    FetchByMetadataRequest,
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

    def test_fetch_by_metadata_request_with_filter(self):
        request = IndexRequestFactory.fetch_by_metadata_request(filter={"genre": {"$eq": "action"}})
        assert request == FetchByMetadataRequest(filter={"genre": {"$eq": "action"}})

    def test_fetch_by_metadata_request_with_filter_and_namespace(self):
        request = IndexRequestFactory.fetch_by_metadata_request(
            filter={"genre": {"$in": ["comedy", "drama"]}}, namespace="my_namespace"
        )
        assert request == FetchByMetadataRequest(
            filter={"genre": {"$in": ["comedy", "drama"]}}, namespace="my_namespace"
        )

    def test_fetch_by_metadata_request_with_limit(self):
        request = IndexRequestFactory.fetch_by_metadata_request(
            filter={"year": {"$gte": 2020}}, limit=50
        )
        assert request == FetchByMetadataRequest(filter={"year": {"$gte": 2020}}, limit=50)

    def test_fetch_by_metadata_request_with_pagination_token(self):
        request = IndexRequestFactory.fetch_by_metadata_request(
            filter={"status": "active"}, pagination_token="token123"
        )
        assert request == FetchByMetadataRequest(
            filter={"status": "active"}, pagination_token="token123"
        )

    def test_fetch_by_metadata_request_with_all_params(self):
        request = IndexRequestFactory.fetch_by_metadata_request(
            filter={"genre": {"$eq": "action"}, "year": {"$eq": 2020}},
            namespace="my_namespace",
            limit=100,
            pagination_token="token456",
        )
        assert request == FetchByMetadataRequest(
            filter={"genre": {"$eq": "action"}, "year": {"$eq": 2020}},
            namespace="my_namespace",
            limit=100,
            pagination_token="token456",
        )

    def test_fetch_by_metadata_request_without_optional_params(self):
        request = IndexRequestFactory.fetch_by_metadata_request(filter={"genre": {"$eq": "action"}})
        assert request.filter == {"genre": {"$eq": "action"}}
        assert request.namespace is None
        assert request.limit is None
        assert request.pagination_token is None

    # region: update request tests

    def test_update_request_with_filter(self):
        request = IndexRequestFactory.update_request(id="vec1", filter={"genre": {"$eq": "action"}})
        assert request.id == "vec1"
        assert request.filter == {"genre": {"$eq": "action"}}

    def test_update_request_with_filter_and_set_metadata(self):
        request = IndexRequestFactory.update_request(
            id="vec1", set_metadata={"status": "active"}, filter={"genre": {"$eq": "drama"}}
        )
        assert request.id == "vec1"
        assert request.set_metadata == {"status": "active"}
        assert request.filter == {"genre": {"$eq": "drama"}}

    def test_update_request_with_filter_and_values(self):
        values = [0.1, 0.2, 0.3]
        request = IndexRequestFactory.update_request(
            id="vec1", values=values, filter={"year": {"$gte": 2020}}
        )
        assert request.id == "vec1"
        assert request.values == values
        assert request.filter == {"year": {"$gte": 2020}}

    def test_update_request_with_filter_and_namespace(self):
        request = IndexRequestFactory.update_request(
            id="vec1", filter={"status": "active"}, namespace="my_namespace"
        )
        assert request.id == "vec1"
        assert request.filter == {"status": "active"}
        assert request.namespace == "my_namespace"

    def test_update_request_with_filter_and_sparse_values(self):
        sparse_values = {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]}
        request = IndexRequestFactory.update_request(
            id="vec1", sparse_values=sparse_values, filter={"genre": {"$in": ["action", "comedy"]}}
        )
        assert request.id == "vec1"
        assert request.sparse_values is not None
        assert request.filter == {"genre": {"$in": ["action", "comedy"]}}

    def test_update_request_with_all_params_including_filter(self):
        values = [0.1, 0.2, 0.3]
        set_metadata = {"status": "active", "updated": True}
        sparse_values = {"indices": [1, 2], "values": [0.4, 0.5]}
        filter_dict = {"genre": {"$eq": "action"}, "year": {"$gte": 2020}}
        request = IndexRequestFactory.update_request(
            id="vec1",
            values=values,
            set_metadata=set_metadata,
            namespace="my_namespace",
            sparse_values=sparse_values,
            filter=filter_dict,
        )
        assert request.id == "vec1"
        assert request.values == values
        assert request.set_metadata == set_metadata
        assert request.namespace == "my_namespace"
        assert request.sparse_values is not None
        assert request.filter == filter_dict

    def test_update_request_without_filter_backward_compatibility(self):
        """Test that update_request still works without filter parameter (backward compatibility)."""
        request = IndexRequestFactory.update_request(
            id="vec1", values=[0.1, 0.2, 0.3], namespace="ns"
        )
        assert request.id == "vec1"
        assert request.values == [0.1, 0.2, 0.3]
        assert request.namespace == "ns"
        # Filter should not be set when not provided
        assert not hasattr(request, "filter") or request.filter is None

    def test_update_request_with_filter_only_no_id(self):
        """Test update_request with filter only (no id) for bulk updates."""
        request = IndexRequestFactory.update_request(
            filter={"genre": {"$eq": "action"}}, set_metadata={"status": "active"}
        )
        assert request.filter == {"genre": {"$eq": "action"}}
        assert request.set_metadata == {"status": "active"}
        # id should not be set when not provided
        assert not hasattr(request, "id") or request.id is None

    def test_update_request_with_id_only_no_filter(self):
        """Test update_request with id only (no filter) - backward compatibility."""
        request = IndexRequestFactory.update_request(id="vec1", values=[0.1, 0.2, 0.3])
        assert request.id == "vec1"
        assert request.values == [0.1, 0.2, 0.3]
        # Filter should not be set when not provided
        assert not hasattr(request, "filter") or request.filter is None

    def test_update_request_with_simple_equality_filter(self):
        """Test update_request with simple equality filter."""
        request = IndexRequestFactory.update_request(id="vec1", filter={"genre": "action"})
        assert request.id == "vec1"
        assert request.filter == {"genre": "action"}

    def test_update_request_with_filter_operators(self):
        """Test update_request with various filter operators."""
        # Test $in operator
        request1 = IndexRequestFactory.update_request(
            id="vec1", filter={"genre": {"$in": ["action", "comedy", "drama"]}}
        )
        assert request1.filter == {"genre": {"$in": ["action", "comedy", "drama"]}}

        # Test $gte operator
        request2 = IndexRequestFactory.update_request(id="vec1", filter={"year": {"$gte": 2020}})
        assert request2.filter == {"year": {"$gte": 2020}}

        # Test $lte operator
        request3 = IndexRequestFactory.update_request(id="vec1", filter={"rating": {"$lte": 4.5}})
        assert request3.filter == {"rating": {"$lte": 4.5}}

        # Test $ne operator
        request4 = IndexRequestFactory.update_request(
            id="vec1", filter={"status": {"$ne": "deleted"}}
        )
        assert request4.filter == {"status": {"$ne": "deleted"}}

    def test_update_request_with_complex_nested_filter(self):
        """Test update_request with complex nested filters using $and and $or."""
        complex_filter = {
            "$or": [
                {"$and": [{"genre": "drama"}, {"year": {"$gte": 2020}}]},
                {"$and": [{"genre": "comedy"}, {"year": {"$lt": 2000}}]},
            ]
        }
        request = IndexRequestFactory.update_request(id="vec1", filter=complex_filter)
        assert request.id == "vec1"
        assert request.filter == complex_filter

    def test_update_request_with_dry_run(self):
        """Test update_request with dry_run parameter."""
        request = IndexRequestFactory.update_request(
            filter={"genre": {"$eq": "action"}}, dry_run=True
        )
        assert request.filter == {"genre": {"$eq": "action"}}
        assert request.dry_run is True

    def test_update_request_with_dry_run_false(self):
        """Test update_request with dry_run=False."""
        request = IndexRequestFactory.update_request(
            filter={"genre": {"$eq": "action"}}, dry_run=False
        )
        assert request.filter == {"genre": {"$eq": "action"}}
        assert request.dry_run is False

    def test_update_request_with_dry_run_and_set_metadata(self):
        """Test update_request with dry_run and set_metadata."""
        request = IndexRequestFactory.update_request(
            filter={"genre": {"$eq": "drama"}}, set_metadata={"status": "active"}, dry_run=True
        )
        assert request.filter == {"genre": {"$eq": "drama"}}
        assert request.set_metadata == {"status": "active"}
        assert request.dry_run is True

    def test_update_request_with_dry_run_and_all_params(self):
        """Test update_request with dry_run and all parameters."""
        values = [0.1, 0.2, 0.3]
        set_metadata = {"status": "active"}
        sparse_values = {"indices": [1, 2], "values": [0.4, 0.5]}
        filter_dict = {"genre": {"$eq": "action"}}
        request = IndexRequestFactory.update_request(
            values=values,
            set_metadata=set_metadata,
            namespace="my_namespace",
            sparse_values=sparse_values,
            filter=filter_dict,
            dry_run=True,
        )
        assert request.values == values
        assert request.set_metadata == set_metadata
        assert request.namespace == "my_namespace"
        assert request.sparse_values is not None
        assert request.filter == filter_dict
        assert request.dry_run is True

    def test_update_request_without_dry_run_not_included(self):
        """Test that dry_run is not included in request when not provided."""
        request = IndexRequestFactory.update_request(
            filter={"genre": {"$eq": "action"}}, set_metadata={"status": "active"}
        )
        assert request.filter == {"genre": {"$eq": "action"}}
        assert request.set_metadata == {"status": "active"}
        # dry_run should not be set when not provided
        # Since parse_non_empty_args filters out None values, dry_run won't be in _data_store
        assert "dry_run" not in request._data_store

    # endregion
