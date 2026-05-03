"""Priority-7 smoke test — async pod index + collections roundtrip.

Mirror of ``test_pod_collections_sync.py`` against ``AsyncPinecone``.

Pod indexes are slow even compared to serverless — set timeouts generously.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from pinecone import AsyncPinecone, NotFoundError, PodSpec
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    async_cleanup_resource,
    async_ensure_index_deleted,
    async_poll_until,
    unique_name,
)
from tests.smoke.helpers import async_wait_for_vector_count


async def _async_wait_for_collection_deletion(
    pc: AsyncPinecone, name: str, *, timeout: int = 300, interval: int = 5
) -> None:
    """Async equivalent: poll until collection is no longer listed."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            cols = await pc.collections.list()
            if name not in cols.names():
                return
        except NotFoundError:
            return
        except Exception as exc:
            print(f"  collections.list() failed during cleanup wait: {exc}")
        await asyncio.sleep(interval)
    print(f"  WARNING: collection {name} still present after {timeout}s")


POD_ENV = "us-east-1-aws"
DIM = 8
NAMESPACE = "smoke-pod-ns-async"


def _vec(i: int) -> dict[str, object]:
    base = (i + 1) * 0.05
    return {
        "id": f"p{i}",
        "values": [base + j * 0.01 for j in range(DIM)],
    }


async def _async_wait_for_pod_warmup(
    idx: object,
    ping_id: str,
    *,
    namespace: str,
    timeout: int = 180,
    interval: int = 3,
) -> None:
    """Async equivalent of helpers.wait_for_pod_warmup."""
    start = time.monotonic()
    last_exc: Exception | None = None
    while time.monotonic() - start < timeout:
        try:
            await idx.fetch(ids=[ping_id], namespace=namespace)  # type: ignore[attr-defined]
            return
        except Exception as exc:
            last_exc = exc
        await asyncio.sleep(interval)
    raise TimeoutError(
        f"Pod index did not warm up within {timeout}s (last fetch error: {last_exc!r})"
    )


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_pod_collections_smoke_async(api_key: str) -> None:
    pc = AsyncPinecone(api_key=api_key)
    pod1_name = unique_name(f"{SMOKE_PREFIX}-pod1-async")
    collection_name = unique_name(f"{SMOKE_PREFIX}-col-async")
    pod2_name = unique_name(f"{SMOKE_PREFIX}-pod2-async")

    try:
        await pc.indexes.create(
            name=pod1_name,
            spec=PodSpec(environment=POD_ENV, pod_type="p1.x1", pods=1),
            dimension=DIM,
            metric="cosine",
            timeout=600,
        )

        idx1 = await pc.index(name=pod1_name)
        try:
            await idx1.upsert(vectors=[_vec(0)], namespace=NAMESPACE)
            await _async_wait_for_pod_warmup(idx1, "p0", namespace=NAMESPACE, timeout=600)

            await idx1.upsert(vectors=[_vec(i) for i in range(1, 6)], namespace=NAMESPACE)
            await async_wait_for_vector_count(idx1, NAMESPACE, expected=6, timeout=120)

            q = await idx1.query(
                top_k=3,
                vector=[0.10 + j * 0.01 for j in range(DIM)],
                namespace=NAMESPACE,
            )
            assert len(q.matches) == 3
        finally:
            await idx1.close()

        await pc.indexes.configure(pod1_name, replicas=2)

        await pc.collections.create(name=collection_name, source=pod1_name)

        async def col_status() -> object:
            return await pc.collections.describe(collection_name)

        await async_poll_until(
            col_status,
            lambda c: getattr(c, "status", None) == "Ready",
            timeout=600,
            interval=10,
            description=f"async collection {collection_name}",
        )

        cols = await pc.collections.list()
        assert collection_name in cols.names()

        await pc.indexes.create(
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

        idx2 = await pc.index(name=pod2_name)
        try:
            await _async_wait_for_pod_warmup(idx2, "p0", namespace=NAMESPACE, timeout=300)
            stats = await idx2.describe_index_stats()
            ns = stats.namespaces.get(NAMESPACE)
            assert ns is not None
            assert ns.vector_count >= 1
        finally:
            await idx2.close()
    finally:
        # See cleanup-ordering note in test_pod_collections_sync.py.
        await async_ensure_index_deleted(pc, pod2_name, timeout=600)
        await async_cleanup_resource(
            lambda: pc.collections.delete(collection_name),
            collection_name,
            "collection",
        )
        await _async_wait_for_collection_deletion(pc, collection_name, timeout=300)
        await async_ensure_index_deleted(pc, pod1_name, timeout=600)
        await pc.close()
