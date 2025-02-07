import pytest
import os
from pinecone import Vector, SparseValues
from ..helpers import poll_stats_for_namespace, embedding_values


@pytest.mark.skipif(
    os.getenv("METRIC") != "dotproduct", reason="Only metric=dotprodouct indexes support hybrid"
)
class TestUpsertHybrid:
    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_upsert_to_namespace_with_sparse_embedding_values(
        self, idx, namespace, use_nondefault_namespace
    ):
        target_namespace = namespace if use_nondefault_namespace else ""

        # Upsert with sparse values object
        idx.upsert(
            vectors=[
                Vector(
                    id="1",
                    values=embedding_values(),
                    sparse_values=SparseValues(indices=[0, 1], values=embedding_values()),
                )
            ],
            namespace=target_namespace,
        )

        # Upsert with sparse values dict
        idx.upsert(
            vectors=[
                {
                    "id": "2",
                    "values": embedding_values(),
                    "sparse_values": {"indices": [0, 1], "values": embedding_values()},
                },
                {
                    "id": "3",
                    "values": embedding_values(),
                    "sparse_values": {"indices": [0, 1], "values": embedding_values()},
                },
            ],
            namespace=target_namespace,
        )

        poll_stats_for_namespace(idx, target_namespace, 9)

        # Check the vector count reflects some data has been upserted
        stats = idx.describe_index_stats()
        assert stats.total_vector_count >= 9
        assert stats.namespaces[target_namespace].vector_count == 9
