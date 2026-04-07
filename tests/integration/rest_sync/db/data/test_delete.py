import pytest
from tests.integration.helpers import poll_until_lsn_reconciled, embedding_values, random_string


@pytest.fixture(scope="session")
def delete_by_ids_namespace():
    return random_string(20)


@pytest.fixture(scope="session")
def delete_all_namespace():
    return random_string(20)


class TestDelete:
    def test_delete_by_ids(self, idx, delete_by_ids_namespace):
        ids = [f"del-{i}" for i in range(3)]
        vectors = [(id, embedding_values()) for id in ids]
        upsert_resp = idx.upsert(vectors=vectors, namespace=delete_by_ids_namespace)
        poll_until_lsn_reconciled(
            idx, upsert_resp._response_info, namespace=delete_by_ids_namespace
        )

        delete_resp = idx.delete(ids=ids[:2], namespace=delete_by_ids_namespace)
        assert delete_resp is None or isinstance(delete_resp, dict)

        fetched = idx.fetch(ids=ids, namespace=delete_by_ids_namespace)
        assert ids[2] in fetched.vectors
        assert ids[0] not in fetched.vectors
        assert ids[1] not in fetched.vectors

    def test_delete_all(self, idx, delete_all_namespace):
        vectors = [(f"delall-{i}", embedding_values()) for i in range(3)]
        upsert_resp = idx.upsert(vectors=vectors, namespace=delete_all_namespace)
        poll_until_lsn_reconciled(
            idx, upsert_resp._response_info, namespace=delete_all_namespace
        )

        delete_resp = idx.delete(namespace=delete_all_namespace, delete_all=True)
        assert delete_resp is None or isinstance(delete_resp, dict)
