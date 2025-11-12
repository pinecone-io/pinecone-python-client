import logging
import pytest
import random
from tests.integration.helpers import embedding_values, random_string, poll_until_lsn_reconciled

from pinecone import Vector, SparseValues

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fetch_sparse_namespace():
    return random_string(20)


def seed_sparse(sparse_idx, namespace):
    upsert1 = sparse_idx.upsert(
        vectors=[
            Vector(
                id=str(i),
                sparse_values=SparseValues(
                    indices=[i, random.randint(2000, 4000)], values=embedding_values(2)
                ),
                metadata={"genre": "action", "runtime": 120},
            )
            for i in range(50)
        ],
        namespace=namespace,
    )
    return upsert1._response_info


@pytest.fixture(scope="function")
def seed_for_fetch_sparse(sparse_idx, fetch_sparse_namespace):
    response_info1 = seed_sparse(sparse_idx, fetch_sparse_namespace)
    response_info2 = seed_sparse(sparse_idx, "__default__")

    poll_until_lsn_reconciled(sparse_idx, response_info1, namespace=fetch_sparse_namespace)
    poll_until_lsn_reconciled(sparse_idx, response_info2, namespace="__default__")
    yield


@pytest.mark.usefixtures("seed_for_fetch_sparse")
class TestFetchSparse:
    def test_fetch_sparse_index(self, sparse_idx):
        fetch_results = sparse_idx.fetch(ids=[str(i) for i in range(10)])
        assert fetch_results.namespace == ""
        assert len(fetch_results.vectors) == 10
        for i in range(10):
            logger.debug(fetch_results.vectors[str(i)])
            assert fetch_results.vectors[str(i)].id == str(i)
            assert fetch_results.vectors[str(i)].sparse_values is not None
            assert len(fetch_results.vectors[str(i)].sparse_values.indices) == 2
            assert len(fetch_results.vectors[str(i)].sparse_values.values) == 2
            assert fetch_results.vectors[str(i)].metadata is not None
            assert fetch_results.vectors[str(i)].metadata["genre"] == "action"
            assert fetch_results.vectors[str(i)].metadata["runtime"] == 120
