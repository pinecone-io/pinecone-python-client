"""Opt-in smoke test that exercises retry behavior against the real Pinecone API.

Gated behind PINECONE_RETRY_SMOKE=1. Creates a legacy (vectors-API) index,
drives a high-concurrency upsert until the API rate-limits, and asserts the
operation completes successfully. Costs < $1 per run.

Each lane gets its own dedicated legacy index from
:func:`tests.integration.legacy_index.create_legacy_index` rather than the
shared :func:`legacy_index_factory` -- these tests drive 100K vectors per
lane and measure per-lane wall-clock throttling, so sharing one index across
lanes would mix their write load. See :mod:`tests.integration.legacy_index`
for why 2026-07 cannot create an index the vectors API will serve at all.

Each lane proves the API pushed back by reading the per-host bulk gate's
``throttle_events`` counter, which both transports increment on every
retryable response -- so a 429, but a 503 or a 504 as well. The
``"Throttled response"`` log line is not a usable signal here: the REST
transport emits it only when it goes on to schedule a retry, so a 429 on the
last attempt or against an exhausted retry budget leaves no record, and the
gRPC data path never emits it at all.

What the gate then does with that signal -- halve the limit, hold for any
``Retry-After`` -- is deterministic local logic, asserted in
:mod:`tests.unit._internal.bulk.test_core_aimd_contract`. These lanes assert
only the part no unit test can reach: a live index pushed back, and the
upsert finished anyway.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator

import pytest

from pinecone import AsyncPinecone, Pinecone
from pinecone._internal.bulk.registry import get_registry
from tests.integration.legacy_index import (
    assert_serves_vectors_api,
    create_legacy_index,
    delete_legacy_index,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("PINECONE_RETRY_SMOKE") != "1",
    reason="Set PINECONE_RETRY_SMOKE=1 to run live retry smoke tests.",
)

# Expected wall-clock for 100K vectors at batch_size=100, max_concurrency=64:
#   Best case (no throttling): ~30s
#   With throttling (this test): 60-180s, dominated by AIMD ramp-up and
#     server-side quota recovery windows.
# The 180s bound is intentionally generous; tighten on stable infra.


@pytest.fixture(scope="module")
def smoke_index_rest(api_key: str, client: Pinecone) -> Generator[str, None, None]:
    """Module-scoped legacy index (dim=1536, cosine) for REST smoke tests."""
    legacy_idx = create_legacy_index(api_key, dimension=1536, metric="cosine")
    assert_serves_vectors_api(client, legacy_idx)
    try:
        yield legacy_idx.name
    finally:
        delete_legacy_index(api_key, legacy_idx.name)


@pytest.fixture(scope="module")
def smoke_index_async(api_key: str, client: Pinecone) -> Generator[str, None, None]:
    """Module-scoped legacy index (dim=1536, cosine) for async smoke tests."""
    legacy_idx = create_legacy_index(api_key, dimension=1536, metric="cosine")
    assert_serves_vectors_api(client, legacy_idx)
    try:
        yield legacy_idx.name
    finally:
        delete_legacy_index(api_key, legacy_idx.name)


@pytest.fixture(scope="module")
def smoke_index_grpc(api_key: str, client: Pinecone) -> Generator[str, None, None]:
    """Module-scoped legacy index (dim=1536, cosine) for gRPC smoke tests."""
    legacy_idx = create_legacy_index(api_key, dimension=1536, metric="cosine")
    assert_serves_vectors_api(client, legacy_idx)
    try:
        yield legacy_idx.name
    finally:
        delete_legacy_index(api_key, legacy_idx.name)


@pytest.mark.integration
def test_high_concurrency_upsert_under_real_throttling(
    client: Pinecone,
    smoke_index_rest: str,
) -> None:
    """Sync REST: a 100K-vector upsert at max_concurrency=64 draws pushback and completes."""
    index = client.index(name=smoke_index_rest)
    vectors: list[tuple[str, list[float]]] = [(f"id-{i}", [0.1] * 1536) for i in range(100_000)]
    gate = get_registry().get(index.host)
    throttle_events_before = gate.throttle_events

    start = time.monotonic()
    index.upsert(vectors=vectors, batch_size=100, max_concurrency=64, show_progress=False)
    elapsed = time.monotonic() - start

    # Operation must complete successfully.
    assert index.describe_index_stats().total_vector_count >= 100_000

    # Must complete in reasonable wall-clock time (3 minutes; serverless quota
    # recovery + AIMD ramp; exact bound is empirical — adjust on first run).
    assert elapsed < 180, f"upsert took {elapsed:.1f}s, expected < 180s"

    # The load must have actually drawn pushback, or the retry path went
    # unexercised.
    assert gate.throttle_events > throttle_events_before, (
        "no retryable response reached the host gate; increase concurrency or batch count"
    )


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_high_concurrency_upsert_under_real_throttling(
    async_client: AsyncPinecone,
    smoke_index_async: str,
) -> None:
    """Async REST: a 100K-vector upsert at max_concurrency=64 draws pushback and completes."""
    index = await async_client.index(name=smoke_index_async)
    vectors: list[tuple[str, list[float]]] = [(f"id-{i}", [0.1] * 1536) for i in range(100_000)]
    gate = get_registry().get(index.host)
    throttle_events_before = gate.throttle_events

    start = time.monotonic()
    await index.upsert(vectors=vectors, batch_size=100, max_concurrency=64, show_progress=False)
    elapsed = time.monotonic() - start

    # Operation must complete successfully.
    stats = await index.describe_index_stats()
    assert stats.total_vector_count >= 100_000

    # Must complete in reasonable wall-clock time (3 minutes; serverless quota
    # recovery + AIMD ramp; exact bound is empirical — adjust on first run).
    assert elapsed < 180, f"upsert took {elapsed:.1f}s, expected < 180s"

    # The load must have actually drawn pushback, or the retry path went
    # unexercised.
    assert gate.throttle_events > throttle_events_before, (
        "no retryable response reached the host gate; increase concurrency or batch count"
    )


@pytest.mark.integration
def test_grpc_high_concurrency_upsert_under_real_throttling(
    client: Pinecone,
    smoke_index_grpc: str,
) -> None:
    """gRPC: a 100K-vector upsert at max_concurrency=64 draws pushback and completes.

    The Rust transport's throttle callback reports into the same process-global
    gate registry the REST transports feed, so this lane asserts on the same
    counter the REST lanes do.
    """
    index = client.index(name=smoke_index_grpc, grpc=True)
    vectors: list[tuple[str, list[float]]] = [(f"id-{i}", [0.1] * 1536) for i in range(100_000)]
    gate = get_registry().get(index.host)
    throttle_events_before = gate.throttle_events

    start = time.monotonic()
    index.upsert(vectors=vectors, batch_size=100, max_concurrency=64, show_progress=False)
    elapsed = time.monotonic() - start

    # Operation must complete successfully.
    assert index.describe_index_stats().total_vector_count >= 100_000

    # Must complete in reasonable wall-clock time (3 minutes; serverless quota
    # recovery + AIMD ramp; exact bound is empirical — adjust on first run).
    assert elapsed < 180, f"upsert took {elapsed:.1f}s, expected < 180s"

    # The load must have actually drawn pushback, or the retry path went
    # unexercised.
    assert gate.throttle_events > throttle_events_before, (
        "no retryable response reached the host gate; increase concurrency or batch count"
    )
