import pytest
from pinecone import Vector, PineconeException
from tests.integration.helpers import poll_until_lsn_reconciled, embedding_values, generate_name


@pytest.fixture(scope="class")
def namespace_query_async(request):
    return generate_name(request.node.name, "upsert-namespace")


@pytest.mark.usefixtures("namespace_query_async")
class TestUpsertWithAsyncReq:
    def test_upsert_to_namespace(self, idx, namespace_query_async):
        target_namespace = namespace_query_async

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

        # Use returned futures
        from concurrent.futures import as_completed

        total_upserted = 0
        upsert_lsn = []
        for future in as_completed([upsert1, upsert2, upsert3], timeout=10):
            total_upserted += future.result().upserted_count
            upsert_lsn.append(future.result()._response_info)
        assert total_upserted == 9

        for response_info in upsert_lsn:
            poll_until_lsn_reconciled(idx, response_info, namespace=target_namespace)

    def test_upsert_to_namespace_when_failed_req(self, idx, namespace_query_async):
        target_namespace = namespace_query_async

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
        upsert_lsn = []
        for future in done:
            if future.exception():
                assert future is upsert2
                assert isinstance(future.exception(), PineconeException)
                assert "Vector dimension 10 does not match the dimension of the index 2" in str(
                    future.exception()
                )
            else:
                total_upserted += future.result().upserted_count
                upsert_lsn.append(future.result()._response_info)
        assert total_upserted == 6

        for response_info in upsert_lsn:
            poll_until_lsn_reconciled(idx, response_info, namespace=target_namespace)
