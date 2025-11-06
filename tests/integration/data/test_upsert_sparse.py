import random
from pinecone import Vector, SparseValues
from ..helpers import embedding_values, random_string, poll_until_lsn_reconciled

import logging

logger = logging.getLogger(__name__)


class TestUpsertSparse:
    def test_upsert_sparse_to_namespace(self, sparse_idx):
        target_namespace = random_string(20)

        # Upsert with objects
        response1 = sparse_idx.upsert(
            vectors=[
                Vector(
                    id=str(i),
                    sparse_values=SparseValues(
                        indices=[i, random.randint(2000, 4000)], values=embedding_values(2)
                    ),
                )
                for i in range(50)
            ],
            namespace=target_namespace,
        )

        # Upsert with dict
        response2 = sparse_idx.upsert(
            vectors=[
                {
                    "id": str(i),
                    "sparse_values": {
                        "indices": [i, random.randint(2000, 4000)],
                        "values": embedding_values(2),
                    },
                }
                for i in range(51, 100)
            ],
            namespace=target_namespace,
        )

        # Upsert with mixed types, dict with SparseValues object
        response3 = sparse_idx.upsert(
            vectors=[
                {
                    "id": str(i),
                    "sparse_values": SparseValues(
                        indices=[i, random.randint(2000, 4000)], values=embedding_values(2)
                    ),
                }
                for i in range(101, 150)
            ],
            namespace=target_namespace,
        )

        # Upsert with mixed types, object with dict
        response4 = sparse_idx.upsert(
            vectors=[
                Vector(
                    id=str(i),
                    sparse_values={
                        "indices": [i, random.randint(2000, 4000)],
                        "values": embedding_values(2),
                    },
                )
                for i in range(151, 200)
            ],
            namespace=target_namespace,
        )

        poll_until_lsn_reconciled(sparse_idx, response1._response_info, namespace=target_namespace)
        poll_until_lsn_reconciled(sparse_idx, response2._response_info, namespace=target_namespace)
        poll_until_lsn_reconciled(sparse_idx, response3._response_info, namespace=target_namespace)
        poll_until_lsn_reconciled(sparse_idx, response4._response_info, namespace=target_namespace)

        results = sparse_idx.query(
            sparse_vector={"indices": [5, 6, 7, 8, 9], "values": embedding_values(5)},
            namespace=target_namespace,
            top_k=5,
        )

        assert len(results.matches) == 5

        # Can query with SparseValues object
        results2 = sparse_idx.query(
            sparse_vector=SparseValues(indices=[5, 6, 7, 8, 9], values=embedding_values(5)),
            namespace=target_namespace,
            top_k=5,
        )
        assert len(results2.matches) == 5
