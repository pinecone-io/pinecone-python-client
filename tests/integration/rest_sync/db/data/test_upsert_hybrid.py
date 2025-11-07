import pytest
from pinecone import Vector, SparseValues
from tests.integration.helpers import poll_until_lsn_reconciled, embedding_values, random_string


class TestUpsertHybrid:
    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_upsert_to_namespace_with_sparse_embedding_values(
        self, hybrid_idx, use_nondefault_namespace
    ):
        target_namespace = random_string(10) if use_nondefault_namespace else None

        # Upsert with sparse values object
        response1 = hybrid_idx.upsert(
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
        response2 = hybrid_idx.upsert(
            vectors=[
                {
                    "id": "2",
                    "values": embedding_values(),
                    "sparse_values": {"indices": [2, 3], "values": embedding_values()},
                },
                {
                    "id": "3",
                    "values": embedding_values(),
                    "sparse_values": {"indices": [4, 5], "values": embedding_values()},
                },
            ],
            namespace=target_namespace,
        )

        poll_until_lsn_reconciled(hybrid_idx, response1._response_info, namespace=target_namespace)
        poll_until_lsn_reconciled(hybrid_idx, response2._response_info, namespace=target_namespace)

        # Fetch the vectors to make sure they were upserted correctly
        fetched_vec = hybrid_idx.fetch(ids=["1", "2", "3"], namespace=target_namespace)
        assert len(fetched_vec.vectors.keys()) == 3
        assert "1" in fetched_vec.vectors
        assert "2" in fetched_vec.vectors
        assert "3" in fetched_vec.vectors

        assert fetched_vec.vectors["1"].sparse_values.indices == [0, 1]
        assert fetched_vec.vectors["2"].sparse_values.indices == [2, 3]
        assert fetched_vec.vectors["3"].sparse_values.indices == [4, 5]
