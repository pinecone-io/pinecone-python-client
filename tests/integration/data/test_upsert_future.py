import pytest
import os
from pinecone import Vector, PineconeException
from ..helpers import poll_stats_for_namespace
from .utils import embedding_values


class TestUpsertWithAsyncReq:
    @pytest.mark.skipif(
        os.getenv("USE_GRPC") != "true", reason="PineconeGrpcFutures only returned from grpc client"
    )
    def test_upsert_to_namespace(self, idx, namespace):
        target_namespace = namespace

        # Upsert with tuples
        upsert1 = idx.upsert(
            vectors=[
                ("1", embedding_values()),
                ("2", embedding_values()),
                ("3", embedding_values()),
            ],
            namespace=target_namespace,
            async_req=True,
        )

        # Upsert with objects
        upsert2 = idx.upsert(
            vectors=[
                Vector(id="4", values=embedding_values()),
                Vector(id="5", values=embedding_values()),
                Vector(id="6", values=embedding_values()),
            ],
            namespace=target_namespace,
            async_req=True,
        )

        # Upsert with dict
        upsert3 = idx.upsert(
            vectors=[
                {"id": "7", "values": embedding_values()},
                {"id": "8", "values": embedding_values()},
                {"id": "9", "values": embedding_values()},
            ],
            namespace=target_namespace,
            async_req=True,
        )

        poll_stats_for_namespace(idx, target_namespace, 9)

        # Check the vector count reflects some data has been upserted
        stats = idx.describe_index_stats()
        assert stats.total_vector_count >= 9
        assert stats.namespaces[target_namespace].vector_count == 9

        # Use returned futures
        from concurrent.futures import as_completed

        total_upserted = 0
        for future in as_completed([upsert1, upsert2, upsert3], timeout=10):
            total_upserted += future.result().upserted_count

        assert total_upserted == 9

    @pytest.mark.skipif(
        os.getenv("USE_GRPC") != "true", reason="PineconeGrpcFutures only returned from grpc client"
    )
    def test_upsert_to_namespace_when_failed_req(self, idx, namespace):
        target_namespace = namespace

        # Upsert with tuples
        upsert1 = idx.upsert(
            vectors=[
                ("1", embedding_values()),
                ("2", embedding_values()),
                ("3", embedding_values()),
            ],
            namespace=target_namespace,
            async_req=True,
        )

        # Upsert with objects
        wrong_dimension = 10
        upsert2 = idx.upsert(
            vectors=[
                Vector(id="4", values=embedding_values(wrong_dimension)),
                Vector(id="5", values=embedding_values(wrong_dimension)),
                Vector(id="6", values=embedding_values(wrong_dimension)),
            ],
            namespace=target_namespace,
            async_req=True,
        )

        # Upsert with dict
        upsert3 = idx.upsert(
            vectors=[
                {"id": "7", "values": embedding_values()},
                {"id": "8", "values": embedding_values()},
                {"id": "9", "values": embedding_values()},
            ],
            namespace=target_namespace,
            async_req=True,
        )

        from concurrent.futures import wait, ALL_COMPLETED

        done, not_done = wait([upsert1, upsert2, upsert3], timeout=10, return_when=ALL_COMPLETED)

        assert len(done) == 3
        assert len(not_done) == 0

        total_upserted = 0
        for future in done:
            if future.exception():
                assert future is upsert2
                assert isinstance(future.exception(), PineconeException)
                assert "Vector dimension 10 does not match the dimension of the index 2" in str(
                    future.exception()
                )
            else:
                total_upserted += future.result().upserted_count
        assert total_upserted == 6
