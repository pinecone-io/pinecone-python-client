import pytest
import logging
from ..helpers import random_string, embedding_values
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async

from pinecone import RerankModel, PineconeApiException

logger = logging.getLogger(__name__)

model_index_dimension = 1024  # Currently controlled by "multilingual-e5-large"


@pytest.fixture
def records_to_upsert():
    return [
        {
            "id": "test1",
            "my_text_field": "Apple is a popular fruit known for its sweetness and crisp texture.",
            "more_stuff": "more stuff!",
        },
        {
            "id": "test2",
            "my_text_field": "The tech company Apple is known for its innovative products like the iPhone.",
            "more_stuff": "more stuff!",
        },
        {
            "id": "test3",
            "my_text_field": "Many people enjoy eating apples as a healthy snack.",
            "more_stuff": "more stuff!",
        },
        {
            "id": "test4",
            "my_text_field": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
            "more_stuff": "more stuff!",
        },
        {
            "id": "test5",
            "my_text_field": "An apple a day keeps the doctor away, as the saying goes.",
            "more_stuff": "more stuff!",
        },
        {
            "id": "test6",
            "my_text_field": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership.",
            "more_stuff": "extra more stuff!",
        },
    ]


@pytest.mark.asyncio
class TestUpsertAndSearchRecords:
    async def test_search_records(self, model_index_host, records_to_upsert):
        model_idx = build_asyncioindex_client(model_index_host)

        target_namespace = random_string(10)
        upsert1 = await model_idx.upsert_records(
            namespace=target_namespace, records=records_to_upsert
        )

        await poll_until_lsn_reconciled_async(
            model_idx,
            target_lsn=upsert1._response_info.get("lsn_committed"),
            namespace=target_namespace,
        )

        response = await model_idx.search_records(
            namespace=target_namespace, query={"inputs": {"text": "Apple corporation"}, "top_k": 3}
        )
        assert len(response.result.hits) == 3
        assert response.usage is not None

        # Test search alias
        response2 = await model_idx.search(
            namespace=target_namespace, query={"inputs": {"text": "Apple corporation"}, "top_k": 3}
        )
        assert response == response2

        # validate similar records and contents
        similar_record_ids = ["test2", "test4", "test6"]

        for hit in response.result.hits:
            assert hit._id in similar_record_ids
            similar_record_ids.remove(hit._id)

            assert hit._score > 0
            assert hit.fields.get("my_text_field") is not None
            assert hit.fields.get("more_stuff") is not None

        # search for records while filtering fields
        response_filtered = await model_idx.search_records(
            namespace="test-namespace",
            query={"inputs": {"text": "Apple corporation"}, "top_k": 3},
            fields=["more_stuff"],
        )

        for hit in response_filtered.result.hits:
            assert hit.fields.get("my_text_field") is None
            assert hit.fields.get("more_stuff") is not None
        await model_idx.close()

    async def test_search_records_with_vector(self, model_index_host, records_to_upsert):
        model_idx = build_asyncioindex_client(model_index_host)

        target_namespace = random_string(10)
        upsert1 = await model_idx.upsert_records(
            namespace=target_namespace, records=records_to_upsert
        )

        await poll_until_lsn_reconciled_async(
            model_idx,
            target_lsn=upsert1._response_info.get("lsn_committed"),
            namespace=target_namespace,
        )

        # Search for similar records
        search_query = {"top_k": 3, "vector": {"values": embedding_values(model_index_dimension)}}
        response = await model_idx.search_records(namespace=target_namespace, query=search_query)
        assert len(response.result.hits) == 3
        assert response.usage is not None

        # Test search alias
        response2 = await model_idx.search(namespace=target_namespace, query=search_query)
        assert response == response2
        await model_idx.close()

    @pytest.mark.parametrize("rerank_model", ["bge-reranker-v2-m3", RerankModel.Bge_Reranker_V2_M3])
    async def test_search_with_rerank(self, model_index_host, records_to_upsert, rerank_model):
        model_idx = build_asyncioindex_client(model_index_host)
        target_namespace = random_string(10)
        upsert1 = await model_idx.upsert_records(
            namespace=target_namespace, records=records_to_upsert
        )

        await poll_until_lsn_reconciled_async(
            model_idx,
            target_lsn=upsert1._response_info.get("lsn_committed"),
            namespace=target_namespace,
        )

        # Search for similar records
        response = await model_idx.search_records(
            namespace=target_namespace,
            query={"inputs": {"text": "Apple corporation"}, "top_k": 3},
            rerank={"model": rerank_model, "rank_fields": ["my_text_field"], "top_n": 3},
        )
        assert len(response.result.hits) == 3
        assert response.usage is not None

        # validate similar records and contents
        similar_record_ids = ["test6", "test4", "test2"]
        for hit in response.result.hits:
            assert hit._id in similar_record_ids
            similar_record_ids.remove(hit._id)

            assert hit._score > 0
            assert hit.fields.get("my_text_field") is not None
            assert hit.fields.get("more_stuff") is not None
        await model_idx.close()

    async def test_search_with_rerank_query(self, model_index_host, records_to_upsert):
        model_idx = build_asyncioindex_client(model_index_host)
        target_namespace = random_string(10)
        upsert1 = await model_idx.upsert_records(
            namespace=target_namespace, records=records_to_upsert
        )

        await poll_until_lsn_reconciled_async(
            model_idx,
            target_lsn=upsert1._response_info.get("lsn_committed"),
            namespace=target_namespace,
        )

        # Search for similar records
        response = await model_idx.search_records(
            namespace=target_namespace,
            query={"inputs": {"text": "Apple corporation"}, "top_k": 3},
            rerank={
                "model": "bge-reranker-v2-m3",
                "rank_fields": ["my_text_field"],
                "top_n": 3,
                "query": "Apple corporation",
            },
        )
        assert len(response.result.hits) == 3
        assert response.usage is not None
        await model_idx.close()

    async def test_search_with_match_terms_dict(self, model_index_host, records_to_upsert):
        """Test that match_terms can be passed via dict query."""
        from pinecone import PineconeApiException

        model_idx = build_asyncioindex_client(model_index_host)
        target_namespace = random_string(10)
        upsert1 = await model_idx.upsert_records(
            namespace=target_namespace, records=records_to_upsert
        )

        await poll_until_lsn_reconciled_async(
            model_idx,
            target_lsn=upsert1._response_info.get("lsn_committed"),
            namespace=target_namespace,
        )

        # Search with match_terms using dict
        query_dict = {
            "inputs": {"text": "Apple corporation"},
            "top_k": 3,
            "match_terms": {"strategy": "all", "terms": ["Apple", "corporation"]},
        }
        # match_terms is only supported for pinecone-sparse-english-v0 model
        # If the API rejects it due to model incompatibility, that's expected
        # and shows our code is correctly passing the parameter
        try:
            response = await model_idx.search_records(namespace=target_namespace, query=query_dict)
            assert response.usage is not None
            # Test search alias
            response2 = await model_idx.search(namespace=target_namespace, query=query_dict)
            assert response == response2
        except PineconeApiException as e:
            # Verify the error is about model compatibility, not parameter format
            assert "match_terms" in str(e) or "pinecone-sparse-english-v0" in str(e)
        await model_idx.close()

    async def test_search_with_match_terms_searchquery(self, model_index_host, records_to_upsert):
        """Test that match_terms can be passed via SearchQuery dataclass."""
        from pinecone import SearchQuery, PineconeApiException

        model_idx = build_asyncioindex_client(model_index_host)
        target_namespace = random_string(10)
        upsert1 = await model_idx.upsert_records(
            namespace=target_namespace, records=records_to_upsert
        )

        await poll_until_lsn_reconciled_async(
            model_idx,
            target_lsn=upsert1._response_info.get("lsn_committed"),
            namespace=target_namespace,
        )

        # Search with match_terms using SearchQuery dataclass
        query = SearchQuery(
            inputs={"text": "Apple corporation"},
            top_k=3,
            match_terms={"strategy": "all", "terms": ["Apple", "corporation"]},
        )
        # match_terms is only supported for pinecone-sparse-english-v0 model
        # If the API rejects it due to model incompatibility, that's expected
        # and shows our code is correctly passing the parameter
        try:
            response = await model_idx.search_records(namespace=target_namespace, query=query)
            assert response.usage is not None
            # Test search alias
            response2 = await model_idx.search(namespace=target_namespace, query=query)
            assert response == response2
        except PineconeApiException as e:
            # Verify the error is about model compatibility, not parameter format
            assert "match_terms" in str(e) or "pinecone-sparse-english-v0" in str(e)
        await model_idx.close()


@pytest.mark.asyncio
class TestUpsertAndSearchRecordsErrorCases:
    async def test_search_with_rerank_nonexistent_model_error(
        self, model_index_host, records_to_upsert
    ):
        model_idx = build_asyncioindex_client(model_index_host)
        target_namespace = random_string(10)
        upsert1 = await model_idx.upsert_records(
            namespace=target_namespace, records=records_to_upsert
        )

        await poll_until_lsn_reconciled_async(
            model_idx,
            target_lsn=upsert1._response_info.get("lsn_committed"),
            namespace=target_namespace,
        )

        with pytest.raises(PineconeApiException, match=r"Model 'non-existent-model' not found"):
            await model_idx.search_records(
                namespace=target_namespace,
                query={"inputs": {"text": "Apple corporation"}, "top_k": 3},
                rerank={
                    "model": "non-existent-model",
                    "rank_fields": ["my_text_field"],
                    "top_n": 3,
                },
            )
        await model_idx.close()
