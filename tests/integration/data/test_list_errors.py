from pinecone import PineconeException
import pytest
from ..helpers import poll_fetch_for_ids_in_namespace, random_string, embedding_values
import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def list_errors_namespace():
    return random_string(10)


@pytest.fixture(scope="session")
def seed_for_list2(idx, list_errors_namespace, wait=True):
    logger.debug(f"Upserting into list namespace '{list_errors_namespace}'")
    for i in range(0, 1000, 50):
        idx.upsert(
            vectors=[(str(i + d), embedding_values(2)) for d in range(50)],
            namespace=list_errors_namespace,
        )

    if wait:
        poll_fetch_for_ids_in_namespace(idx, ids=["999"], namespace=list_errors_namespace)

    yield


@pytest.mark.usefixtures("seed_for_list2")
class TestListErrors:
    @pytest.mark.skip(reason="Bug filed https://github.com/pinecone-io/pinecone-db/issues/9578")
    def test_list_change_prefix_while_fetching_next_page(self, idx, list_errors_namespace):
        results = idx.list_paginated(prefix="99", limit=5, namespace=list_errors_namespace)
        with pytest.raises(PineconeException) as e:
            results = idx.list_paginated(
                prefix="98", limit=5, pagination_token=results.pagination.next
            )
            print(results)
        assert "prefix" in str(e.value)

    @pytest.mark.skip(reason="Bug filed")
    def test_list_change_namespace_while_fetching_next_page(self, idx, list_errors_namespace):
        results = idx.list_paginated(limit=5, namespace=list_errors_namespace)
        with pytest.raises(PineconeException) as e:
            idx.list_paginated(
                limit=5, namespace="new-namespace", pagination_token=results.pagination.next
            )
        assert "namespace" in str(e.value)
