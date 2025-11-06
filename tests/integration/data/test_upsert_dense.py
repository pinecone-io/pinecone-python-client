import pytest
from pinecone import Vector
from ..helpers import poll_until_lsn_reconciled, embedding_values, random_string


@pytest.fixture(scope="session")
def upsert_dense_namespace():
    return random_string(10)


class TestUpsertDense:
    def test_upsert_to_namespace(self, idx, upsert_dense_namespace):
        target_namespace = upsert_dense_namespace

        # Upsert with tuples
        idx.upsert(
            vectors=[
                ("1", embedding_values()),
                ("2", embedding_values()),
                ("3", embedding_values()),
            ],
            namespace=target_namespace,
        )

        # Upsert with objects
        idx.upsert(
            vectors=[
                Vector(id="4", values=embedding_values()),
                Vector(id="5", values=embedding_values()),
                Vector(id="6", values=embedding_values()),
            ],
            namespace=target_namespace,
        )

        # Upsert with dict
        response3 = idx.upsert(
            vectors=[
                {"id": "7", "values": embedding_values()},
                {"id": "8", "values": embedding_values()},
                {"id": "9", "values": embedding_values()},
            ],
            namespace=target_namespace,
        )

        poll_until_lsn_reconciled(
            idx, response3._response_info.get("lsn_committed"), operation_name="upsert_to_namespace"
        )

        stats = idx.describe_index_stats()
        assert stats.namespaces[target_namespace].vector_count == 9
