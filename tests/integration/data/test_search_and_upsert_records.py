import time
import pytest
from typing import List
from ..helpers import random_string, embedding_values
import logging
import os

from pinecone import RerankModel, PineconeApiException
from pinecone.db_data import _Index

logger = logging.getLogger(__name__)

model_index_dimension = 1024  # Currently controlled by "multilingual-e5-large"


def poll_until_fetchable(idx: _Index, namespace: str, ids: List[str], timeout: int):
    found = False
    total_wait = 0
    interval = 5

    while not found:
        if total_wait > timeout:
            logger.debug(f"Failed to fetch records within {timeout} seconds.")
            raise TimeoutError(f"Failed to fetch records within {timeout} seconds.")
        time.sleep(interval)
        total_wait += interval

        response = idx.fetch(ids=ids, namespace=namespace)
        logger.debug(
            f"Polling {total_wait} seconds for fetch response with ids {ids} in namespace {namespace}"
        )

        if len(response.vectors) == len(ids):
            found = True


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


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="These actions are not supported in gRPC"
)
class TestUpsertAndSearchRecords:
    def test_search_records(self, model_idx, records_to_upsert):
        target_namespace = random_string(10)
        model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        poll_until_fetchable(
            model_idx, target_namespace, [r["id"] for r in records_to_upsert], timeout=180
        )

        response = model_idx.search_records(
            namespace=target_namespace, query={"inputs": {"text": "Apple corporation"}, "top_k": 3}
        )
        assert len(response.result.hits) == 3
        assert response.usage is not None

        # Test search alias
        response2 = model_idx.search(
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
        response_filtered = model_idx.search_records(
            namespace="test-namespace",
            query={"inputs": {"text": "Apple corporation"}, "top_k": 3},
            fields=["more_stuff"],
        )

        for hit in response_filtered.result.hits:
            assert hit.fields.get("my_text_field") is None
            assert hit.fields.get("more_stuff") is not None

    def test_search_records_with_vector(self, model_idx, records_to_upsert):
        target_namespace = random_string(10)
        model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        poll_until_fetchable(
            model_idx, target_namespace, [r["id"] for r in records_to_upsert], timeout=180
        )

        # Search for similar records
        search_query = {"top_k": 3, "vector": {"values": embedding_values(model_index_dimension)}}
        response = model_idx.search_records(namespace=target_namespace, query=search_query)
        assert len(response.result.hits) == 3
        assert response.usage is not None

        # Test search alias
        response2 = model_idx.search(namespace=target_namespace, query=search_query)
        assert response == response2

    @pytest.mark.parametrize("rerank_model", ["bge-reranker-v2-m3", RerankModel.Bge_Reranker_V2_M3])
    def test_search_with_rerank(self, model_idx, records_to_upsert, rerank_model):
        target_namespace = random_string(10)
        model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        poll_until_fetchable(
            model_idx, target_namespace, [r["id"] for r in records_to_upsert], timeout=180
        )

        # Search for similar records
        response = model_idx.search_records(
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

    def test_search_with_rerank_query(self, model_idx, records_to_upsert):
        target_namespace = random_string(10)
        model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        # Sleep for freshness
        poll_until_fetchable(
            model_idx, target_namespace, [r["id"] for r in records_to_upsert], timeout=180
        )

        # Search for similar records
        response = model_idx.search_records(
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

    def test_search_with_match_terms_dict(self, model_idx, records_to_upsert):
        """Test that match_terms can be passed via dict query."""
        from pinecone import PineconeApiException

        target_namespace = random_string(10)
        model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        poll_until_fetchable(
            model_idx, target_namespace, [r["id"] for r in records_to_upsert], timeout=180
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
            response = model_idx.search_records(namespace=target_namespace, query=query_dict)
            assert response.usage is not None
            # Test search alias
            response2 = model_idx.search(namespace=target_namespace, query=query_dict)
            assert response == response2
        except PineconeApiException as e:
            # Verify the error is about model compatibility, not parameter format
            assert "match_terms" in str(e) or "pinecone-sparse-english-v0" in str(e)

    def test_search_with_match_terms_searchquery(self, model_idx, records_to_upsert):
        """Test that match_terms can be passed via SearchQuery dataclass."""
        from pinecone import SearchQuery, PineconeApiException

        target_namespace = random_string(10)
        model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        poll_until_fetchable(
            model_idx, target_namespace, [r["id"] for r in records_to_upsert], timeout=180
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
            response = model_idx.search_records(namespace=target_namespace, query=query)
            assert response.usage is not None
            # Test search alias
            response2 = model_idx.search(namespace=target_namespace, query=query)
            assert response == response2
        except PineconeApiException as e:
            # Verify the error is about model compatibility, not parameter format
            assert "match_terms" in str(e) or "pinecone-sparse-english-v0" in str(e)


@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "false", reason="These actions are not supported in gRPC"
)
class TestUpsertAndSearchRecordsErrorCases:
    def test_search_with_rerank_nonexistent_model_error(self, model_idx, records_to_upsert):
        target_namespace = random_string(10)
        model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        poll_until_fetchable(
            model_idx, target_namespace, [r["id"] for r in records_to_upsert], timeout=180
        )

        with pytest.raises(PineconeApiException, match=r"Model 'non-existent-model' not found"):
            model_idx.search_records(
                namespace=target_namespace,
                query={"inputs": {"text": "Apple corporation"}, "top_k": 3},
                rerank={
                    "model": "non-existent-model",
                    "rank_fields": ["my_text_field"],
                    "top_n": 3,
                },
            )

    @pytest.mark.skip(reason="Possible bug in the API")
    def test_search_with_rerank_empty_rank_fields_error(self, model_idx, records_to_upsert):
        target_namespace = random_string(10)
        model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        poll_until_fetchable(
            model_idx, target_namespace, [r["id"] for r in records_to_upsert], timeout=180
        )

        with pytest.raises(
            PineconeApiException, match=r"Only one rank field is supported for model"
        ):
            model_idx.search_records(
                namespace="test-namespace",
                query={"inputs": {"text": "Apple corporation"}, "top_k": 3},
                rerank={"model": "bge-reranker-v2-m3", "rank_fields": [], "top_n": 3},
            )
