from pinecone import Vector
from tests.integration.helpers import poll_until_lsn_reconciled, random_string
import logging
import time

logger = logging.getLogger(__name__)


def seed_vectors(idx, namespace):
    logger.info("Seeding vectors with ids [id1, id2, id3] to namespace '%s'", namespace)
    response = idx.upsert(
        vectors=[
            Vector(id="id1", values=[0.1, 0.2]),
            Vector(id="id2", values=[0.1, 0.2]),
            Vector(id="id3", values=[0.1, 0.2]),
        ],
        namespace=namespace,
    )
    poll_until_lsn_reconciled(idx, response._response_info, namespace=namespace)


class TestDeleteFuture:
    def test_delete_future(self, idx):
        namespace = random_string(10)

        seed_vectors(idx, namespace)

        delete_one = idx.delete(ids=["id1"], namespace=namespace, async_req=True)
        delete_two = idx.delete(ids=["id2"], namespace=namespace, async_req=True)

        from concurrent.futures import as_completed

        for future in as_completed([delete_one, delete_two], timeout=10):
            resp = future.result()
            assert resp["_response_info"] is not None

        time.sleep(10)

        # Verify that the vectors are deleted
        from concurrent.futures import wait, ALL_COMPLETED

        fetch_results = idx.fetch(ids=["id1", "id2"], namespace=namespace, async_req=True)
        done, not_done = wait([fetch_results], timeout=10, return_when=ALL_COMPLETED)

        assert len(done) == 1
        assert len(not_done) == 0
        results = fetch_results.result()
        assert len(results.vectors) == 0

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
            assert resp["_response_info"] is not None
