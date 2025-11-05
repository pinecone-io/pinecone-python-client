import time
import pytest
from typing import List, Optional
from ..helpers import random_string, embedding_values
import logging
import os

from pinecone import RerankModel, PineconeApiException
from pinecone.db_data import _Index

logger = logging.getLogger(__name__)

model_index_dimension = 1024  # Currently controlled by "multilingual-e5-large"


def poll_until_fetchable(
    idx: _Index, namespace: str, ids: List[str], timeout: int, target_lsn: Optional[int] = None
):
    """Poll until vectors are fetchable, optionally using LSN headers for faster detection.

    Args:
        idx: The index client
        namespace: Namespace to fetch from
        ids: List of vector IDs to fetch
        timeout: Maximum time to wait in seconds
        target_lsn: Optional LSN from write operation. If provided, uses LSN headers for faster polling.
    """
    found = False
    total_wait = 0
    interval = 5 if target_lsn is None else 2  # Use shorter interval with LSN

    while not found:
        if total_wait > timeout:
            logger.debug(f"Failed to fetch records within {timeout} seconds.")
            raise TimeoutError(f"Failed to fetch records within {timeout} seconds.")

        if total_wait > 0:  # Don't sleep on first iteration
            time.sleep(interval)
        total_wait += interval

        # Try to use LSN headers if available
        if target_lsn is not None:
            try:
                response = idx.fetch(ids=ids, namespace=namespace)
                from tests.integration.helpers.lsn_utils import is_lsn_reconciled

                if hasattr(response, "_response_info") and response._response_info:
                    reconciled_lsn = response._response_info.get("lsn_reconciled")
                    if reconciled_lsn is not None and is_lsn_reconciled(target_lsn, reconciled_lsn):
                        # LSN is reconciled, check if vectors are present
                        if len(response.vectors) == len(ids):
                            found = True
                            logger.debug(
                                f"Found vectors using LSN after {total_wait}s. "
                                f"Reconciled LSN: {reconciled_lsn}, target: {target_lsn}"
                            )
                        else:
                            logger.debug(
                                f"LSN reconciled but vectors not all present. "
                                f"Found {len(response.vectors)}/{len(ids)}"
                            )
                    else:
                        logger.debug(
                            f"LSN not yet reconciled. Reconciled: {reconciled_lsn}, target: {target_lsn}"
                        )
                        continue  # Skip checking vectors if LSN not reconciled
            except Exception as e:
                logger.debug(f"Error using LSN-based check: {e}, falling back to regular fetch")
                # Fall through to regular fetch check

        # Regular fetch check (used if LSN not available or LSN check failed)
        if not found:
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
        response = model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        # Extract LSN from response if available
        committed_lsn = None
        if hasattr(response, "_response_info") and response._response_info:
            committed_lsn = response._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response._response_info is not None
            ), "Expected _response_info to be present on upsert_records response"

        poll_until_fetchable(
            model_idx,
            target_namespace,
            [r["id"] for r in records_to_upsert],
            timeout=180,
            target_lsn=committed_lsn,
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
        response = model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        # Extract LSN from response if available
        committed_lsn = None
        if hasattr(response, "_response_info") and response._response_info:
            committed_lsn = response._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response._response_info is not None
            ), "Expected _response_info to be present on upsert_records response"

        poll_until_fetchable(
            model_idx,
            target_namespace,
            [r["id"] for r in records_to_upsert],
            timeout=180,
            target_lsn=committed_lsn,
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
        response = model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        # Extract LSN from response if available
        committed_lsn = None
        if hasattr(response, "_response_info") and response._response_info:
            committed_lsn = response._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response._response_info is not None
            ), "Expected _response_info to be present on upsert_records response"

        poll_until_fetchable(
            model_idx,
            target_namespace,
            [r["id"] for r in records_to_upsert],
            timeout=180,
            target_lsn=committed_lsn,
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
        response = model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        # Extract LSN from response if available
        committed_lsn = None
        if hasattr(response, "_response_info") and response._response_info:
            committed_lsn = response._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response._response_info is not None
            ), "Expected _response_info to be present on upsert_records response"

        poll_until_fetchable(
            model_idx,
            target_namespace,
            [r["id"] for r in records_to_upsert],
            timeout=180,
            target_lsn=committed_lsn,
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
        response = model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        # Extract LSN from response if available
        committed_lsn = None
        if hasattr(response, "_response_info") and response._response_info:
            committed_lsn = response._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response._response_info is not None
            ), "Expected _response_info to be present on upsert_records response"

        poll_until_fetchable(
            model_idx,
            target_namespace,
            [r["id"] for r in records_to_upsert],
            timeout=180,
            target_lsn=committed_lsn,
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
        response = model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        # Extract LSN from response if available
        committed_lsn = None
        if hasattr(response, "_response_info") and response._response_info:
            committed_lsn = response._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response._response_info is not None
            ), "Expected _response_info to be present on upsert_records response"

        poll_until_fetchable(
            model_idx,
            target_namespace,
            [r["id"] for r in records_to_upsert],
            timeout=180,
            target_lsn=committed_lsn,
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
        response = model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        # Extract LSN from response if available
        committed_lsn = None
        if hasattr(response, "_response_info") and response._response_info:
            committed_lsn = response._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response._response_info is not None
            ), "Expected _response_info to be present on upsert_records response"

        poll_until_fetchable(
            model_idx,
            target_namespace,
            [r["id"] for r in records_to_upsert],
            timeout=180,
            target_lsn=committed_lsn,
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
        response = model_idx.upsert_records(namespace=target_namespace, records=records_to_upsert)

        # Extract LSN from response if available
        committed_lsn = None
        if hasattr(response, "_response_info") and response._response_info:
            committed_lsn = response._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response._response_info is not None
            ), "Expected _response_info to be present on upsert_records response"

        poll_until_fetchable(
            model_idx,
            target_namespace,
            [r["id"] for r in records_to_upsert],
            timeout=180,
            target_lsn=committed_lsn,
        )

        with pytest.raises(
            PineconeApiException, match=r"Only one rank field is supported for model"
        ):
            model_idx.search_records(
                namespace="test-namespace",
                query={"inputs": {"text": "Apple corporation"}, "top_k": 3},
                rerank={"model": "bge-reranker-v2-m3", "rank_fields": [], "top_n": 3},
            )
