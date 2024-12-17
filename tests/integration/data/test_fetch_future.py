import os
import pytest
from ..helpers import poll_fetch_for_ids_in_namespace
from .utils import embedding_values
from pinecone import Vector
import logging

logger = logging.getLogger(__name__)

if os.environ.get("USE_GRPC") == "true":
    from pinecone.grpc import PineconeGrpcFuture


@pytest.fixture(scope="class")
def seed_for_fetch(idx, namespace):
    # Upsert without metadata
    logger.info("Seeding vectors without metadata")
    idx.upsert(
        vectors=[
            ("1", embedding_values(2)),
            ("2", embedding_values(2)),
            ("3", embedding_values(2)),
        ],
        namespace=namespace,
    )

    # Upsert with metadata
    logger.info("Seeding vectors with metadata")
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
    idx.upsert(
        vectors=[
            {"id": "7", "values": embedding_values(2)},
            {"id": "8", "values": embedding_values(2)},
            {"id": "9", "values": embedding_values(2)},
        ],
        namespace=namespace,
    )

    poll_fetch_for_ids_in_namespace(
        idx, ids=["1", "2", "3", "4", "5", "6", "7", "8", "9"], namespace=namespace
    )
    yield


@pytest.mark.usefixtures("seed_for_fetch")
@pytest.mark.skipif(
    os.getenv("USE_GRPC") != "true", reason="PineconeGrpcFutures only returned from grpc client"
)
class TestFetchFuture:
    def setup_method(self):
        self.expected_dimension = 2

    def test_fetch_multiple_by_id(self, idx, namespace):
        target_namespace = namespace

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

    def test_fetch_single_by_id(self, idx, namespace):
        target_namespace = namespace

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

    def test_fetch_nonexistent_id(self, idx, namespace):
        target_namespace = namespace

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
