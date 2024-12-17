import pytest
from pinecone import Vector
from ..helpers import poll_stats_for_namespace, embedding_values, random_string


@pytest.fixture(scope="session")
def upsert_dense_namespace():
    return random_string(10)


class TestUpsertDense:
    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_upsert_to_namespace(self, idx, upsert_dense_namespace, use_nondefault_namespace):
        target_namespace = upsert_dense_namespace if use_nondefault_namespace else ""

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
        idx.upsert(
            vectors=[
                {"id": "7", "values": embedding_values()},
                {"id": "8", "values": embedding_values()},
                {"id": "9", "values": embedding_values()},
            ],
            namespace=target_namespace,
        )

        poll_stats_for_namespace(idx, target_namespace, 9)

        # Check the vector count reflects some data has been upserted
        stats = idx.describe_index_stats()
        assert stats.total_vector_count >= 9
        assert stats.namespaces[target_namespace].vector_count == 9
