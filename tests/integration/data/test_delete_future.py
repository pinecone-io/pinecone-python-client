import os
import pytest
from pinecone import Vector
from ..helpers import poll_stats_for_namespace, random_string

if os.environ.get("USE_GRPC") == "true":
    from pinecone.grpc import GRPCDeleteResponse


class TestDeleteFuture:
    @pytest.mark.skipif(
        os.getenv("USE_GRPC") != "true", reason="PineconeGrpcFutures only returned from grpc client"
    )
    def test_delete_future(self, idx):
        namespace = random_string(10)

        idx.upsert(
            vectors=[
                Vector(id="id1", values=[0.1, 0.2]),
                Vector(id="id2", values=[0.1, 0.2]),
                Vector(id="id3", values=[0.1, 0.2]),
            ],
            namespace=namespace,
        )
        poll_stats_for_namespace(idx, namespace, 3)

        delete_one = idx.delete(ids=["id1"], namespace=namespace, async_req=True)
        delete_namespace = idx.delete(namespace=namespace, delete_all=True, async_req=True)

        from concurrent.futures import as_completed

        for future in as_completed([delete_one, delete_namespace], timeout=10):
            resp = future.result()
            assert isinstance(resp, GRPCDeleteResponse)
