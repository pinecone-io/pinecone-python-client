import pytest
import os
from pinecone import Vector, SparseValues
from ..helpers import poll_until_lsn_reconciled, embedding_values


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
        response1 = idx.upsert(
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
        response2 = idx.upsert(
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

        poll_until_lsn_reconciled(
            idx, response1._response_info.get("lsn_committed"), namespace=target_namespace
        )
        poll_until_lsn_reconciled(
            idx, response2._response_info.get("lsn_committed"), namespace=target_namespace
        )

        # Check the vector count reflects some data has been upserted
        stats = idx.describe_index_stats()
        assert stats.total_vector_count >= 9
        # The default namespace may be represented as "" or "__default__" in the API response
        if target_namespace == "":
            namespace_key = "__default__" if "__default__" in stats.namespaces else ""
        else:
            namespace_key = target_namespace
        assert namespace_key in stats.namespaces
        assert stats.namespaces[namespace_key].vector_count == 9
