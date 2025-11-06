import logging
import pytest
import random
from ..helpers import embedding_values, random_string, poll_until_lsn_reconciled

from pinecone import PineconeException, FetchResponse, Vector, SparseValues

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fetch_namespace():
    return random_string(10)


def seed(idx, namespace):
    # Upsert without metadata
    logger.info(f"Seeding vectors without metadata into namespace '{namespace}'")
    idx.upsert(
        vectors=[
            ("1", embedding_values(2)),
            ("2", embedding_values(2)),
            ("3", embedding_values(2)),
        ],
        namespace=namespace,
    )

    # Upsert with metadata
    logger.info(f"Seeding vectors with metadata into namespace '{namespace}'")
    idx.upsert(
        vectors=[
            Vector(
                id="4", values=embedding_values(2), metadata={"genre": "action", "runtime": 120}
            ),
            Vector(id="5", values=embedding_values(2), metadata={"genre": "comedy", "runtime": 90}),
            Vector(
                id="6", values=embedding_values(2), metadata={"genre": "romance", "runtime": 240}
            ),
        ],
        namespace=namespace,
    )

    # Upsert with dict
    upsert3 = idx.upsert(
        vectors=[
            {"id": "7", "values": embedding_values(2)},
            {"id": "8", "values": embedding_values(2)},
            {"id": "9", "values": embedding_values(2)},
        ],
        namespace=namespace,
    )
    return upsert3._response_info.get("lsn_committed")


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
    return upsert1._response_info.get("lsn_committed")


@pytest.fixture(scope="function")
def seed_for_fetch(idx, sparse_idx, fetch_namespace):
    lsn_committed = seed(idx, fetch_namespace)
    lsn_committed2 = seed(idx, "__default__")
    lsn_committed3 = seed_sparse(sparse_idx, fetch_namespace)
    lsn_committed4 = seed_sparse(sparse_idx, "__default__")

    poll_until_lsn_reconciled(idx, target_lsn=lsn_committed, namespace=fetch_namespace)
    poll_until_lsn_reconciled(idx, target_lsn=lsn_committed2, namespace="__default__")
    poll_until_lsn_reconciled(sparse_idx, target_lsn=lsn_committed3, namespace=fetch_namespace)
    poll_until_lsn_reconciled(sparse_idx, target_lsn=lsn_committed4, namespace="__default__")
    yield


@pytest.mark.usefixtures("seed_for_fetch")
class TestFetch:
    def setup_method(self):
        self.expected_dimension = 2

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_fetch_multiple_by_id(self, idx, fetch_namespace, use_nondefault_namespace):
        target_namespace = fetch_namespace if use_nondefault_namespace else "__default__"

        results = idx.fetch(ids=["1", "2", "4"], namespace=target_namespace)
        assert isinstance(results, FetchResponse) == True

        assert results.usage is not None
        assert results.usage["read_units"] is not None
        assert results.usage["read_units"] > 0

        assert results.namespace == target_namespace
        assert len(results.vectors) == 3
        assert results.vectors["1"].id == "1"
        assert results.vectors["2"].id == "2"
        # Metadata included, if set
        assert results.vectors["1"].metadata is None
        assert results.vectors["2"].metadata is None
        assert results.vectors["4"].metadata is not None
        assert results.vectors["4"].metadata["genre"] == "action"
        assert results.vectors["4"].metadata["runtime"] == 120
        # Values included
        assert results.vectors["1"].values is not None
        assert len(results.vectors["1"].values) == self.expected_dimension

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_fetch_single_by_id(self, idx, fetch_namespace, use_nondefault_namespace):
        target_namespace = fetch_namespace if use_nondefault_namespace else "__default__"

        results = idx.fetch(ids=["1"], namespace=target_namespace)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 1
        assert results.vectors["1"].id == "1"
        assert results.vectors["1"].metadata is None
        assert results.vectors["1"].values is not None
        assert len(results.vectors["1"].values) == self.expected_dimension

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_fetch_nonexistent_id(self, idx, fetch_namespace, use_nondefault_namespace):
        target_namespace = fetch_namespace if use_nondefault_namespace else "__default__"

        # Fetch id that is missing
        results = idx.fetch(ids=["100"], namespace=target_namespace)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 0

    def test_fetch_nonexistent_namespace(self, idx):
        target_namespace = "nonexistent-namespace"

        # Fetch from namespace with no vectors
        results = idx.fetch(ids=["1"], namespace=target_namespace)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 0

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_fetch_with_empty_list_of_ids(self, idx, fetch_namespace, use_nondefault_namespace):
        target_namespace = fetch_namespace if use_nondefault_namespace else "__default__"

        # Fetch with empty list of ids
        with pytest.raises(PineconeException) as e:
            idx.fetch(ids=[], namespace=target_namespace)
        assert "ids" in str(e.value).lower()

    def test_fetch_unspecified_namespace(self, idx):
        # Fetch without specifying namespace gives default namespace results
        results = idx.fetch(ids=["1", "4"])
        assert results.namespace == ""
        assert results.vectors["1"].id == "1"
        assert results.vectors["1"].values is not None
        assert results.vectors["4"].metadata is not None

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
