import os
import pytest
from pinecone.grpc import PineconeGrpcFuture


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
