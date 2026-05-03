"""Smoke-test specific helpers.

These helpers complement the shared integration-test helpers in
``tests.integration.conftest``. They cover situations specific to the
notebook-style smoke scenarios:

- Polling vector visibility after upsert (freshness window).
- Defeating the pod-index "Ready-but-not-ready" race.
- Capturing DeprecationWarnings from shim methods.
- Locating the sample files reused from ``tests/integration/``.
"""

from __future__ import annotations

import asyncio
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pinecone.async_client.async_index import AsyncIndex
    from pinecone.grpc import GrpcIndex
    from pinecone.index import Index


# ---------------------------------------------------------------------------
# Sample files (reused from tests/integration/)
# ---------------------------------------------------------------------------

_INTEGRATION_DIR = Path(__file__).resolve().parent.parent / "integration"
SAMPLE_TEXT_FILE: Path = _INTEGRATION_DIR / "tiny_file.txt"
SAMPLE_PDF_SMALL: Path = _INTEGRATION_DIR / "tiny_file.pdf"


# ---------------------------------------------------------------------------
# Vector freshness polling
# ---------------------------------------------------------------------------


def wait_for_vector_count(
    idx: Index | GrpcIndex,
    namespace: str,
    expected: int,
    *,
    timeout: int = 60,
    interval: int = 2,
) -> None:
    """Poll ``describe_index_stats`` until the namespace contains ``expected`` vectors.

    User spec: vectors are typically queryable in <10s, occasionally up to ~60s.
    Anything longer is treated as a bug.
    """
    start = time.monotonic()
    last_count = -1
    while time.monotonic() - start < timeout:
        try:
            stats = idx.describe_index_stats()
            ns_stats = stats.namespaces.get(namespace) if stats.namespaces else None
            last_count = ns_stats.vector_count if ns_stats else 0
            if last_count >= expected:
                return
        except Exception as exc:
            print(f"  describe_index_stats failed during freshness wait: {exc}")
        time.sleep(interval)
    raise TimeoutError(
        f"Namespace {namespace!r} only contained {last_count} vectors after "
        f"{timeout}s (expected ≥{expected})"
    )


async def async_wait_for_vector_count(
    idx: AsyncIndex,
    namespace: str,
    expected: int,
    *,
    timeout: int = 60,
    interval: int = 2,
) -> None:
    """Async version of :func:`wait_for_vector_count`."""
    start = time.monotonic()
    last_count = -1
    while time.monotonic() - start < timeout:
        try:
            stats = await idx.describe_index_stats()
            ns_stats = stats.namespaces.get(namespace) if stats.namespaces else None
            last_count = ns_stats.vector_count if ns_stats else 0
            if last_count >= expected:
                return
        except Exception as exc:
            print(f"  describe_index_stats failed during freshness wait: {exc}")
        await asyncio.sleep(interval)
    raise TimeoutError(
        f"Namespace {namespace!r} only contained {last_count} vectors after "
        f"{timeout}s (expected ≥{expected})"
    )


# ---------------------------------------------------------------------------
# Pod-index warmup polling
# ---------------------------------------------------------------------------


def wait_for_pod_warmup(
    idx: Index,
    ping_id: str,
    *,
    namespace: str = "",
    timeout: int = 120,
    interval: int = 3,
) -> None:
    """Retry ``fetch`` until a pod index actually serves data after ``status=Ready``.

    Pod indexes occasionally report Ready a beat before they accept traffic.
    This helper polls a known vector ID until a fetch call succeeds without
    raising, then returns. Used only by the pod/collections scenario.
    """
    start = time.monotonic()
    last_exc: Exception | None = None
    while time.monotonic() - start < timeout:
        try:
            idx.fetch(ids=[ping_id], namespace=namespace)
            return
        except Exception as exc:
            last_exc = exc
        time.sleep(interval)
    raise TimeoutError(
        f"Pod index did not warm up within {timeout}s "
        f"(last fetch error: {last_exc!r})"
    )


# ---------------------------------------------------------------------------
# Deprecation warning capture
# ---------------------------------------------------------------------------


@contextmanager
def capture_deprecation_warning(expected_substring: str = "") -> Iterator[list[Any]]:
    """Context manager that captures DeprecationWarning emissions.

    Usage::

        with capture_deprecation_warning("create_index") as records:
            pc.create_index(...)
        assert records, "expected a DeprecationWarning"

    If ``expected_substring`` is provided, asserts that at least one captured
    warning's message contains it.
    """
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", DeprecationWarning)
        yield records
    relevant = [r for r in records if issubclass(r.category, DeprecationWarning)]
    if not relevant:
        raise AssertionError("Expected at least one DeprecationWarning, got none")
    if expected_substring and not any(
        expected_substring in str(r.message) for r in relevant
    ):
        messages = [str(r.message) for r in relevant]
        raise AssertionError(
            f"No captured DeprecationWarning contained {expected_substring!r}. "
            f"Got: {messages}"
        )
