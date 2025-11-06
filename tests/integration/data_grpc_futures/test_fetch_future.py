import pytest
from ..helpers import poll_until_lsn_reconciled, embedding_values, generate_name
from pinecone import Vector
import logging
from pinecone.grpc import PineconeGrpcFuture

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fetch_namespace_future():
    return generate_name("TestFetchFuture", "fetch-namespace")


def seed(idx, namespace):
    # Upsert without metadata
    logger.info("Seeding vectors without metadata to namespace '%s'", namespace)
    upsert1 = idx.upsert(
        vectors=[
            ("1", embedding_values(2)),
            ("2", embedding_values(2)),
            ("3", embedding_values(2)),
        ],
        namespace=namespace,
    )

    # Upsert with metadata
    logger.info("Seeding vectors with metadata to namespace '%s'", namespace)
    upsert2 = idx.upsert(
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

    poll_until_lsn_reconciled(idx, upsert1._response_info.get("lsn_committed"), namespace=namespace)
    poll_until_lsn_reconciled(idx, upsert2._response_info.get("lsn_committed"), namespace=namespace)
    poll_until_lsn_reconciled(idx, upsert3._response_info.get("lsn_committed"), namespace=namespace)


@pytest.mark.usefixtures("fetch_namespace_future")
@pytest.fixture(scope="class")
def seed_for_fetch(idx, fetch_namespace_future):
    seed(idx, fetch_namespace_future)
    seed(idx, "")
    yield


@pytest.mark.usefixtures("seed_for_fetch")
class TestFetchFuture:
    def setup_method(self):
        self.expected_dimension = 2

    def test_fetch_multiple_by_id(self, idx, fetch_namespace_future):
        target_namespace = fetch_namespace_future

        results = idx.fetch(ids=["1", "2", "4"], namespace=target_namespace, async_req=True)
        assert isinstance(results, PineconeGrpcFuture)

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([results], return_when=FIRST_COMPLETED)

        results = done.pop().result()
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

    def test_fetch_single_by_id(self, idx, fetch_namespace_future):
        target_namespace = fetch_namespace_future

        future = idx.fetch(ids=["1"], namespace=target_namespace, async_req=True)

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([future], return_when=FIRST_COMPLETED)
        results = done.pop().result()

        assert results.namespace == target_namespace
        assert len(results.vectors) == 1
        assert results.vectors["1"].id == "1"
        assert results.vectors["1"].metadata is None
        assert results.vectors["1"].values is not None
        assert len(results.vectors["1"].values) == self.expected_dimension

    def test_fetch_nonexistent_id(self, idx, fetch_namespace_future):
        target_namespace = fetch_namespace_future

        # Fetch id that is missing
        future = idx.fetch(ids=["100"], namespace=target_namespace, async_req=True)

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([future], return_when=FIRST_COMPLETED)
        results = done.pop().result()

        assert results.namespace == target_namespace
        assert len(results.vectors) == 0

    def test_fetch_nonexistent_namespace(self, idx):
        target_namespace = "nonexistent-namespace"

        # Fetch from namespace with no vectors
        future = idx.fetch(ids=["1"], namespace=target_namespace, async_req=True)

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([future], return_when=FIRST_COMPLETED)
        results = done.pop().result()

        assert results.namespace == target_namespace
        assert len(results.vectors) == 0

    def test_fetch_unspecified_namespace(self, idx):
        # Fetch without specifying namespace gives default namespace results
        future = idx.fetch(ids=["1", "4"], async_req=True)

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([future], return_when=FIRST_COMPLETED)
        results = done.pop().result()

        assert results.namespace == ""
        assert results.vectors["1"].id == "1"
        assert results.vectors["1"].values is not None
        assert results.vectors["4"].metadata is not None
