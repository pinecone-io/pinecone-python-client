import pytest
import random
from pinecone import Vector, SparseValues
from ..helpers import poll_stats_for_namespace
from .utils import embedding_values

import logging

logger = logging.getLogger(__name__)


class TestUpsertSparse:
    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_upsert_sparse_to_namespace(self, sparse_idx, use_nondefault_namespace, namespace):
        target_namespace = namespace if use_nondefault_namespace else ""

        # Upsert with objects
        sparse_idx.upsert(
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
        sparse_idx.upsert(
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
        sparse_idx.upsert(
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
        sparse_idx.upsert(
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

        poll_stats_for_namespace(sparse_idx, target_namespace, 99, max_sleep=300)

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
