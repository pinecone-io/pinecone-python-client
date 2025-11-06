import pytest
from ..helpers import poll_until_lsn_reconciled, embedding_values, generate_name
from pinecone import Vector
import logging
from pinecone.grpc import PineconeGrpcFuture

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fetch_by_metadata_namespace_future():
    return generate_name("TestFetchByMetadataFuture", "fetch-by-metadata-namespace")


def seed_for_fetch_by_metadata(idx, namespace):
    # Upsert vectors with different metadata for filtering tests
    logger.info("Seeding vectors with metadata to namespace '%s'", namespace)
    response = idx.upsert(
        vectors=[
            Vector(
                id="meta1", values=embedding_values(2), metadata={"genre": "action", "year": 2020}
            ),
            Vector(
                id="meta2", values=embedding_values(2), metadata={"genre": "comedy", "year": 2021}
            ),
            Vector(
                id="meta3", values=embedding_values(2), metadata={"genre": "action", "year": 2022}
            ),
            Vector(
                id="meta4", values=embedding_values(2), metadata={"genre": "drama", "year": 2020}
            ),
            Vector(
                id="meta5", values=embedding_values(2), metadata={"genre": "action", "year": 2021}
            ),
        ],
        namespace=namespace,
    )

    poll_until_lsn_reconciled(
        idx, response._response_info.get("lsn_committed"), namespace=namespace
    )


@pytest.mark.usefixtures("fetch_by_metadata_namespace_future")
@pytest.fixture(scope="class")
def seed_for_fetch_by_metadata_future(idx, fetch_by_metadata_namespace_future):
    seed_for_fetch_by_metadata(idx, fetch_by_metadata_namespace_future)
    seed_for_fetch_by_metadata(idx, "")
    yield


@pytest.mark.usefixtures("seed_for_fetch_by_metadata_future")
class TestFetchByMetadataFuture:
    def setup_method(self):
        self.expected_dimension = 2

    def test_fetch_by_metadata_simple_filter(self, idx, fetch_by_metadata_namespace_future):
        target_namespace = fetch_by_metadata_namespace_future

        future = idx.fetch_by_metadata(
            filter={"genre": {"$eq": "action"}}, namespace=target_namespace, async_req=True
        )
        assert isinstance(future, PineconeGrpcFuture)

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([future], return_when=FIRST_COMPLETED)
        results = done.pop().result()

        assert results.usage is not None
        assert results.usage["read_units"] is not None
        assert results.usage["read_units"] > 0

        assert results.namespace == target_namespace
        assert len(results.vectors) == 3
        assert "meta1" in results.vectors
        assert "meta3" in results.vectors
        assert "meta5" in results.vectors
        assert results.vectors["meta1"].metadata["genre"] == "action"
        assert results.vectors["meta1"].values is not None
        assert len(results.vectors["meta1"].values) == self.expected_dimension

    def test_fetch_by_metadata_with_limit(self, idx, fetch_by_metadata_namespace_future):
        target_namespace = fetch_by_metadata_namespace_future

        future = idx.fetch_by_metadata(
            filter={"genre": {"$eq": "action"}}, namespace=target_namespace, limit=2, async_req=True
        )

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([future], return_when=FIRST_COMPLETED)
        results = done.pop().result()

        assert results.namespace == target_namespace
        assert len(results.vectors) <= 2

    def test_fetch_by_metadata_with_complex_filter(self, idx, fetch_by_metadata_namespace_future):
        target_namespace = fetch_by_metadata_namespace_future

        future = idx.fetch_by_metadata(
            filter={"genre": {"$eq": "action"}, "year": {"$eq": 2020}},
            namespace=target_namespace,
            async_req=True,
        )

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([future], return_when=FIRST_COMPLETED)
        results = done.pop().result()

        assert results.namespace == target_namespace
        assert len(results.vectors) == 1
        assert "meta1" in results.vectors
        assert results.vectors["meta1"].metadata["genre"] == "action"
        assert results.vectors["meta1"].metadata["year"] == 2020

    def test_fetch_by_metadata_with_in_operator(self, idx, fetch_by_metadata_namespace_future):
        target_namespace = fetch_by_metadata_namespace_future

        future = idx.fetch_by_metadata(
            filter={"genre": {"$in": ["comedy", "drama"]}},
            namespace=target_namespace,
            async_req=True,
        )

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([future], return_when=FIRST_COMPLETED)
        results = done.pop().result()

        assert results.namespace == target_namespace
        assert len(results.vectors) == 2
        assert "meta2" in results.vectors
        assert "meta4" in results.vectors

    def test_fetch_by_metadata_no_results(self, idx, fetch_by_metadata_namespace_future):
        target_namespace = fetch_by_metadata_namespace_future

        future = idx.fetch_by_metadata(
            filter={"genre": {"$eq": "horror"}}, namespace=target_namespace, async_req=True
        )

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([future], return_when=FIRST_COMPLETED)
        results = done.pop().result()

        assert results.namespace == target_namespace
        assert len(results.vectors) == 0

    def test_fetch_by_metadata_unspecified_namespace(self, idx):
        # Fetch from default namespace
        future = idx.fetch_by_metadata(filter={"genre": {"$eq": "action"}}, async_req=True)

        from concurrent.futures import wait, FIRST_COMPLETED

        done, _ = wait([future], return_when=FIRST_COMPLETED)
        results = done.pop().result()

        assert results.namespace == ""
        assert len(results.vectors) == 3
        assert "meta1" in results.vectors
        assert "meta3" in results.vectors
        assert "meta5" in results.vectors
