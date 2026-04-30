"""Priority-7 smoke test — sync pod index + collections roundtrip.

Heaviest scenario: creates a pod-based index, upserts data, snapshots into a
collection, then creates a *new* pod index from that collection. Pod indexes
take longer to provision than serverless and occasionally report ``Ready``
before fetch traffic actually works — handled by ``wait_for_pod_warmup``.

Punchlist coverage (sync):

- pc.indexes.create with PodSpec (incl. source_collection on second create)
- pc.indexes.configure for pod indexes (replicas)
- pc.collections.create / describe / get / list / delete
- Index data plane on pod indexes (upsert, query, fetch, describe_index_stats)
"""

from __future__ import annotations

import time

import pytest

from pinecone import NotFoundError, Pinecone, PodSpec
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    cleanup_resource,
    ensure_index_deleted,
    poll_until,
    unique_name,
)
from tests.smoke.helpers import wait_for_pod_warmup, wait_for_vector_count


def _wait_for_collection_deletion(
    client: Pinecone, name: str, *, timeout: int = 300, interval: int = 5
) -> None:
    """Poll until the collection name is no longer present in the listing."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            if name not in client.collections.list().names():
                return
        except NotFoundError:
            return
        except Exception as exc:
            print(f"  collections.list() failed during cleanup wait: {exc}")
        time.sleep(interval)
    print(f"  WARNING: collection {name} still present after {timeout}s")

POD_ENV = "us-east-1-aws"
DIM = 8
NAMESPACE = "smoke-pod-ns"


def _vec(i: int) -> dict[str, object]:
    base = (i + 1) * 0.05
    return {
        "id": f"p{i}",
        "values": [base + j * 0.01 for j in range(DIM)],
    }


@pytest.mark.smoke
def test_pod_collections_smoke(client: Pinecone) -> None:
    """Pod -> upsert -> collection -> pod-from-collection."""
    pod1_name = unique_name(f"{SMOKE_PREFIX}-pod1")
    collection_name = unique_name(f"{SMOKE_PREFIX}-col")
    pod2_name = unique_name(f"{SMOKE_PREFIX}-pod2")

    try:
        # ----- create pod1 -----
        client.indexes.create(
            name=pod1_name,
            spec=PodSpec(environment=POD_ENV, pod_type="p1.x1", pods=1),
            dimension=DIM,
            metric="cosine",
            timeout=600,
        )

        # ----- attach data plane and warm up -----
        idx1 = client.index(name=pod1_name)
        try:
            # Upsert one ping vector first so wait_for_pod_warmup has
            # something to fetch.
            idx1.upsert(vectors=[_vec(0)], namespace=NAMESPACE)
            wait_for_pod_warmup(idx1, "p0", namespace=NAMESPACE, timeout=600)

            # ----- bulk upsert + freshness -----
            more = [_vec(i) for i in range(1, 6)]
            idx1.upsert(vectors=more, namespace=NAMESPACE)
            wait_for_vector_count(idx1, NAMESPACE, expected=6, timeout=120)

            # ----- query -----
            q = idx1.query(
                top_k=3,
                vector=[0.10 + j * 0.01 for j in range(DIM)],
                namespace=NAMESPACE,
            )
            assert len(q.matches) == 3
        finally:
            idx1.close()

        # ----- configure pod1 (covers configure for pod indexes) -----
        client.indexes.configure(pod1_name, replicas=2)

        # ----- create collection from pod1 -----
        client.collections.create(name=collection_name, source=pod1_name)

        # poll collection until Ready
        poll_until(
            lambda: client.collections.describe(collection_name),
            lambda col: getattr(col, "status", None) == "Ready",
            timeout=600,
            interval=10,
            description=f"collection {collection_name}",
        )

        cols = client.collections.list()
        assert collection_name in cols.names()

        # ----- create pod2 from collection -----
        client.indexes.create(
            name=pod2_name,
            spec=PodSpec(
                environment=POD_ENV,
                pod_type="p1.x1",
                pods=1,
                source_collection=collection_name,
            ),
            dimension=DIM,
            metric="cosine",
            timeout=600,
        )

        idx2 = client.index(name=pod2_name)
        try:
            wait_for_pod_warmup(idx2, "p0", namespace=NAMESPACE, timeout=300)
            stats = idx2.describe_index_stats()
            ns = stats.namespaces.get(NAMESPACE)
            assert ns is not None
            assert ns.vector_count >= 1
        finally:
            idx2.close()
    finally:
        # Best-effort cleanup of every resource. Order:
        #   1. pod2 (no dependencies)
        #   2. collection delete request
        #   3. wait for collection to actually be gone — pod1 cannot be
        #      deleted while a collection that references it is still
        #      pending (server returns 412 FAILED_PRECONDITION)
        #   4. pod1
        ensure_index_deleted(client, pod2_name, timeout=600)
        cleanup_resource(
            lambda: client.collections.delete(collection_name),
            collection_name,
            "collection",
        )
        _wait_for_collection_deletion(client, collection_name, timeout=300)
        ensure_index_deleted(client, pod1_name, timeout=600)
        client.close()
