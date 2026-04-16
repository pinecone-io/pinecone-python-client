"""Generic batch execution engine for parallel bulk operations.

Provides sync (ThreadPoolExecutor) and async (asyncio.Semaphore + gather)
executors that chunk a list of items, run an operation on each chunk in
parallel, collect errors, and optionally display a tqdm progress bar.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, TypeVar

from pinecone.models.batch import BatchError, BatchResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_MAX_WORKERS = 64

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_batch_params(batch_size: int, concurrency: int) -> None:
    """Raise ``ValueError`` for invalid batch_size or concurrency values."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if concurrency < 1 or concurrency > _MAX_WORKERS:
        raise ValueError(f"concurrency must be between 1 and {_MAX_WORKERS}, got {concurrency}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk(items: list[T], size: int) -> list[list[T]]:
    """Split *items* into sublists of at most *size* elements."""
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Progress bar helpers
# ---------------------------------------------------------------------------


class _NoOpProgressBar:
    """Drop-in replacement when tqdm is not installed."""

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix_str(self, s: str) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> _NoOpProgressBar:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _create_progress_bar(
    total: int,
    desc: str,
    show: bool,
) -> Any:
    """Return a tqdm bar if available, otherwise a silent no-op."""
    if not show:
        return _NoOpProgressBar()
    try:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]

        return tqdm(total=total, desc=desc, unit="batch")
    except ImportError:
        return _NoOpProgressBar()


def _empty_result() -> BatchResult:
    """Return a zero-count result for empty input."""
    return BatchResult(
        total_item_count=0,
        successful_item_count=0,
        failed_item_count=0,
        total_batch_count=0,
        successful_batch_count=0,
        failed_batch_count=0,
        errors=[],
    )


# ---------------------------------------------------------------------------
# Sync executor
# ---------------------------------------------------------------------------


def batch_execute(
    *,
    items: list[dict[str, Any]],
    operation: Callable[[list[dict[str, Any]]], Any],
    batch_size: int,
    max_workers: int = 4,
    show_progress: bool = True,
    desc: str = "Batches",
) -> BatchResult:
    """Execute *operation* on *items* in parallel batches.

    Items are split into chunks of *batch_size* and submitted to a
    ``ThreadPoolExecutor`` with *max_workers* threads.  Exceptions raised
    by *operation* are caught per-batch and recorded as ``BatchError``
    entries in the result rather than propagated.

    Args:
        items (list[dict[str, Any]]): Full list of items to process.
        operation (Callable): Callable that accepts a batch (sublist).
        batch_size (int): Maximum items per batch (must be >= 1).
        max_workers (int): Thread pool size for concurrent requests
            (1-64, default 4).
        show_progress (bool): Display a tqdm progress bar when installed.
        desc (str): Label shown on the progress bar.

    Returns:
        BatchResult with aggregated success/failure counts.

    Raises:
        ValueError: If *batch_size* or *max_workers* is out of range.
    """
    _validate_batch_params(batch_size, max_workers)

    if not items:
        return _empty_result()

    batches = _chunk(items, batch_size)
    total_batches = len(batches)
    errors: list[BatchError] = []
    successful_item_count = 0

    progress = _create_progress_bar(total_batches, desc, show_progress)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(operation, batch): (idx, batch) for idx, batch in enumerate(batches)
            }

            for future in as_completed(future_to_batch):
                batch_idx, batch = future_to_batch[future]
                try:
                    future.result()
                    successful_item_count += len(batch)
                except Exception as exc:
                    errors.append(
                        BatchError(
                            batch_index=batch_idx,
                            items=batch,
                            error=exc,
                            error_message=str(exc),
                        )
                    )
                progress.update(1)
    finally:
        progress.close()

    failed_item_count = sum(len(e.items) for e in errors)

    return BatchResult(
        total_item_count=len(items),
        successful_item_count=successful_item_count,
        failed_item_count=failed_item_count,
        total_batch_count=total_batches,
        successful_batch_count=total_batches - len(errors),
        failed_batch_count=len(errors),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Async executor
# ---------------------------------------------------------------------------


async def async_batch_execute(
    *,
    items: list[dict[str, Any]],
    operation: Callable[[list[dict[str, Any]]], Awaitable[Any]],
    batch_size: int,
    max_concurrency: int = 4,
    show_progress: bool = True,
    desc: str = "Batches",
) -> BatchResult:
    """Async version of :func:`batch_execute`.

    Items are split into chunks of *batch_size* and run concurrently
    behind an ``asyncio.Semaphore`` with *max_concurrency* slots.
    Exceptions raised by *operation* are caught per-batch and recorded
    as ``BatchError`` entries in the result rather than propagated.

    Args:
        items (list[dict[str, Any]]): Full list of items to process.
        operation (Callable): Async callable that accepts a batch (sublist).
        batch_size (int): Maximum items per batch (must be >= 1).
        max_concurrency (int): Maximum concurrent batch requests
            (1-64, default 4).
        show_progress (bool): Display a tqdm progress bar when installed.
        desc (str): Label shown on the progress bar.

    Returns:
        BatchResult with aggregated success/failure counts.

    Raises:
        ValueError: If *batch_size* or *max_concurrency* is out of range.
    """
    _validate_batch_params(batch_size, max_concurrency)

    if not items:
        return _empty_result()

    batches = _chunk(items, batch_size)
    total_batches = len(batches)
    errors: list[BatchError] = []
    successful_item_count = 0

    semaphore = asyncio.Semaphore(max_concurrency)
    progress = _create_progress_bar(total_batches, desc, show_progress)

    async def _run_batch(batch_idx: int, batch: list[dict[str, Any]]) -> None:
        # nonlocal is safe: asyncio coroutines run on a single thread,
        # so += and .append() cannot interleave between await points.
        nonlocal successful_item_count
        async with semaphore:
            try:
                await operation(batch)
                successful_item_count += len(batch)
            except Exception as exc:
                errors.append(
                    BatchError(
                        batch_index=batch_idx,
                        items=batch,
                        error=exc,
                        error_message=str(exc),
                    )
                )
            progress.update(1)

    try:
        tasks = [_run_batch(i, batch) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks)
    finally:
        progress.close()

    failed_item_count = sum(len(e.items) for e in errors)

    return BatchResult(
        total_item_count=len(items),
        successful_item_count=successful_item_count,
        failed_item_count=failed_item_count,
        total_batch_count=total_batches,
        successful_batch_count=total_batches - len(errors),
        failed_batch_count=len(errors),
        errors=errors,
    )
