"""Priority-3 smoke test — gRPC dense data plane.

Walks every method on :class:`GrpcIndex` against a serverless dense index.
Includes the future-returning ``*_async`` variants — each one is consumed via
``.result(timeout=10)`` to confirm they return the same shape as their
synchronous counterparts.

Punchlist coverage (gRPC):

- upsert / upsert_async
- query / query_async
- fetch / fetch_async
- delete (ids/delete_all/filter) / delete_async
- update / update_async
- list / list_paginated
- describe_index_stats
- close / ``with`` ctx mgr
- host property

GrpcIndex has no namespace ops, no fetch_by_metadata, no query_namespaces,
no imports. Those checkboxes only get ticked by the sync/async variants.
upsert_from_dataframe is tested in test_upsert_from_dataframe_grpc.py
(pandas-gated).
``upsert_records``, ``search``, and ``search_records`` require an integrated
index — covered by the Priority-4 gRPC test.
"""

from __future__ import annotations

import pytest

from pinecone import GrpcIndex, Pinecone, ServerlessSpec, Vector
from tests.smoke.conftest import SMOKE_PREFIX, ensure_index_deleted, unique_name
from tests.smoke.helpers import wait_for_vector_count

CLOUD = "aws"
REGION = "us-east-1"
DIM = 8


@pytest.mark.smoke
def test_serverless_dense_grpc_smoke(client: Pinecone) -> None:
    """End-to-end gRPC dense data-plane walkthrough."""
    name = unique_name(f"{SMOKE_PREFIX}-srv-dense-grpc")
    try:
        client.indexes.create(
            name=name,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            dimension=DIM,
            metric="cosine",
        )

        # gRPC handle via Pinecone.index(grpc=True)
        idx = client.index(name=name, grpc=True)
        assert isinstance(idx, GrpcIndex)
        assert idx.host

        try:
            # ----- upsert (sync) -----
            # Two metadata categories so we can later delete by filter.
            base_vectors = [
                Vector(
                    id=f"v{i}",
                    values=[0.05 * (i + 1) + j * 0.01 for j in range(DIM)],
                    metadata={"category": "keep" if i < 3 else "drop"},
                )
                for i in range(5)
            ]
            sync_resp = idx.upsert(vectors=base_vectors, namespace="alpha")
            assert sync_resp.upserted_count == 5

            # ----- upsert_async (returns PineconeFuture) -----
            async_vectors = [
                {
                    "id": f"a{i}",
                    "values": [0.30 + 0.01 * i + j * 0.002 for j in range(DIM)],
                }
                for i in range(5)
            ]
            up_future = idx.upsert_async(vectors=async_vectors, namespace="alpha")
            up_async_resp = up_future.result(timeout=10.0)
            assert up_async_resp.upserted_count == 5

            # ----- vector freshness -----
            wait_for_vector_count(idx, "alpha", expected=10, timeout=60)

            # ----- query (sync) -----
            q_resp = idx.query(
                top_k=3,
                vector=[0.10 + j * 0.01 for j in range(DIM)],
                namespace="alpha",
                include_metadata=True,
            )
            assert len(q_resp.matches) == 3

            # ----- query_async -----
            q_future = idx.query_async(
                top_k=2,
                vector=[0.10 + j * 0.01 for j in range(DIM)],
                namespace="alpha",
            )
            q_async = q_future.result(timeout=10.0)
            assert len(q_async.matches) == 2

            # ----- fetch (sync) -----
            fetch_resp = idx.fetch(ids=["v0", "v1"], namespace="alpha")
            assert set(fetch_resp.vectors.keys()) == {"v0", "v1"}

            # ----- fetch_async -----
            f_future = idx.fetch_async(ids=["v2", "v3"], namespace="alpha")
            f_async = f_future.result(timeout=10.0)
            assert set(f_async.vectors.keys()) == {"v2", "v3"}

            # ----- update (sync) -----
            idx.update(
                id="v0",
                set_metadata={"tag": "grpc-updated"},
                namespace="alpha",
            )

            # ----- update_async -----
            u_future = idx.update_async(
                id="v1",
                set_metadata={"tag": "grpc-updated-async"},
                namespace="alpha",
            )
            u_future.result(timeout=10.0)

            # ----- list_paginated -----
            page = idx.list_paginated(prefix="v", limit=5, namespace="alpha")
            ids_seen = [item.id for item in page.vectors]
            assert len(ids_seen) > 0

            # ----- list (iterator) -----
            all_ids: list[str] = []
            for p in idx.list(prefix="v", namespace="alpha"):
                all_ids.extend(item.id for item in p.vectors)
            assert len(all_ids) >= 5

            # ----- describe_index_stats -----
            stats = idx.describe_index_stats()
            assert "alpha" in stats.namespaces

            # ----- delete by filter -----
            # Removes any vector with category=drop. Does not error if zero match.
            idx.delete(filter={"category": {"$eq": "drop"}}, namespace="alpha")

            # ----- delete_all on a throwaway namespace -----
            # Populate a separate namespace, then nuke it. Run against gRPC to
            # exercise the delete_all=True wire path specifically.
            throwaway = [
                Vector(id=f"t{i}", values=[0.50 + i * 0.01 + j * 0.001 for j in range(DIM)])
                for i in range(3)
            ]
            idx.upsert(vectors=throwaway, namespace="gamma-grpc")
            wait_for_vector_count(idx, "gamma-grpc", expected=3, timeout=60)
            idx.delete(delete_all=True, namespace="gamma-grpc")

            # ----- delete (sync) -----
            idx.delete(ids=["v0"], namespace="alpha")

            # ----- delete_async -----
            d_future = idx.delete_async(ids=["v1"], namespace="alpha")
            d_future.result(timeout=10.0)
        finally:
            idx.close()

        # ----- with ctx mgr -----
        with client.index(name=name, grpc=True) as ctx_idx:
            stats2 = ctx_idx.describe_index_stats()
            assert stats2.dimension == DIM
    finally:
        ensure_index_deleted(client, name)
        client.close()
