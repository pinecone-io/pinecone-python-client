import os
import pytest
from pinecone import Vector
from ..helpers import poll_stats_for_namespace, random_string
import logging

logger = logging.getLogger(__name__)

if os.environ.get("USE_GRPC") == "true":
    from pinecone.grpc import GRPCDeleteResponse


def seed_vectors(idx, namespace):
    logger.info("Seeding vectors with ids [id1, id2, id3] to namespace '%s'", namespace)
    idx.upsert(
        vectors=[
            Vector(id="id1", values=[0.1, 0.2]),
            Vector(id="id2", values=[0.1, 0.2]),
            Vector(id="id3", values=[0.1, 0.2]),
        ],
        namespace=namespace,
    )
    poll_stats_for_namespace(idx, namespace, 3)


class TestDeleteFuture:
    @pytest.mark.skipif(
        os.getenv("USE_GRPC") != "true", reason="PineconeGrpcFutures only returned from grpc client"
    )
    def test_delete_future(self, idx):
        namespace = random_string(10)

        seed_vectors(idx, namespace)

        delete_one = idx.delete(ids=["id1"], namespace=namespace, async_req=True)
        delete_two = idx.delete(ids=["id2"], namespace=namespace, async_req=True)

        from concurrent.futures import as_completed

        for future in as_completed([delete_one, delete_two], timeout=10):
            resp = future.result()
            assert isinstance(resp, GRPCDeleteResponse)

    @pytest.mark.skipif(
        os.getenv("USE_GRPC") != "true", reason="PineconeGrpcFutures only returned from grpc client"
    )
    def test_delete_future_by_namespace(self, idx):
        namespace = random_string(10)

        ns1 = f"{namespace}-1"
        ns2 = f"{namespace}-2"

        seed_vectors(idx, ns1)
        seed_vectors(idx, ns2)

        delete_ns1 = idx.delete(namespace=ns1, delete_all=True, async_req=True)
        delete_ns2 = idx.delete(namespace=ns2, delete_all=True, async_req=True)
        from concurrent.futures import as_completed

        for future in as_completed([delete_ns1, delete_ns2], timeout=10):
            resp = future.result()
            assert isinstance(resp, GRPCDeleteResponse)
