"""Unit tests for batch execution helpers."""

from __future__ import annotations

import pytest

from pinecone._internal.batch import (
    _chunk,
    async_batch_execute,
    batch_execute,
)
from pinecone.models.batch import BatchResult

# ---------------------------------------------------------------------------
# Sync: batch_execute
# ---------------------------------------------------------------------------


def test_batch_execute_empty_items() -> None:
    """Empty item list returns zero-count BatchResult."""

    def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    result = batch_execute(items=[], operation=noop, batch_size=10, show_progress=False)

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 0
    assert result.successful_item_count == 0
    assert result.failed_item_count == 0
    assert result.total_batch_count == 0
    assert result.successful_batch_count == 0
    assert result.failed_batch_count == 0
    assert result.errors == []
    assert result.has_errors is False


def test_batch_execute_all_succeed() -> None:
    """10 items with batch_size=3 all succeed — counts match."""
    items = [{"id": str(i)} for i in range(10)]

    def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    result = batch_execute(
        items=items,
        operation=noop,
        batch_size=3,
        max_workers=2,
        show_progress=False,
    )

    assert result.successful_item_count == 10
    assert result.failed_item_count == 0
    assert result.has_errors is False
    assert result.total_item_count == 10


def test_batch_execute_partial_failure() -> None:
    """Second batch raises — 5 succeed, 5 fail, 1 error recorded."""
    items = [{"id": str(i)} for i in range(10)]
    call_count = 0

    def op(batch: list[dict]) -> None:  # type: ignore[type-arg]
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("batch 2 failed")

    result = batch_execute(
        items=items,
        operation=op,
        batch_size=5,
        max_workers=1,  # serial so call order is deterministic
        show_progress=False,
    )

    assert result.successful_item_count == 5
    assert result.failed_item_count == 5
    assert result.has_errors is True
    assert len(result.errors) == 1
    assert result.errors[0].error_message == "batch 2 failed"


def test_batch_execute_invalid_batch_size() -> None:
    """batch_size=0 raises ValueError."""
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        batch_execute(
            items=[{"id": "x"}],
            operation=lambda b: None,
            batch_size=0,
            show_progress=False,
        )


def test_batch_execute_invalid_max_workers_zero() -> None:
    """max_workers=0 raises ValueError."""
    with pytest.raises(ValueError, match="concurrency must be between"):
        batch_execute(
            items=[{"id": "x"}],
            operation=lambda b: None,
            batch_size=1,
            max_workers=0,
            show_progress=False,
        )


def test_batch_execute_invalid_max_workers_too_high() -> None:
    """max_workers=65 raises ValueError."""
    with pytest.raises(ValueError, match="concurrency must be between"):
        batch_execute(
            items=[{"id": "x"}],
            operation=lambda b: None,
            batch_size=1,
            max_workers=65,
            show_progress=False,
        )


# ---------------------------------------------------------------------------
# Async: async_batch_execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_batch_execute_empty_items() -> None:
    """Async: empty item list returns zero-count BatchResult."""

    async def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    result = await async_batch_execute(
        items=[], operation=noop, batch_size=10, show_progress=False
    )

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 0
    assert result.successful_item_count == 0
    assert result.failed_item_count == 0
    assert result.has_errors is False


@pytest.mark.asyncio
async def test_async_batch_execute_all_succeed() -> None:
    """Async: 10 items all succeed — counts match."""
    items = [{"id": str(i)} for i in range(10)]

    async def noop(batch: list[dict]) -> None:  # type: ignore[type-arg]
        pass

    result = await async_batch_execute(
        items=items,
        operation=noop,
        batch_size=3,
        max_concurrency=2,
        show_progress=False,
    )

    assert result.successful_item_count == 10
    assert result.failed_item_count == 0
    assert result.has_errors is False
    assert result.total_item_count == 10


@pytest.mark.asyncio
async def test_async_batch_execute_partial_failure() -> None:
    """Async: one batch raises — failure recorded, others succeed."""
    items = [{"id": str(i)} for i in range(10)]
    failed_batches: list[int] = []

    async def op(batch: list[dict]) -> None:  # type: ignore[type-arg]
        # fail the batch containing id "5"
        if any(item["id"] == "5" for item in batch):
            raise RuntimeError("async batch failed")

    result = await async_batch_execute(
        items=items,
        operation=op,
        batch_size=5,
        max_concurrency=1,
        show_progress=False,
    )

    assert result.has_errors is True
    assert len(result.errors) == 1
    assert result.failed_item_count == 5
    assert result.successful_item_count == 5


# ---------------------------------------------------------------------------
# _chunk helper
# ---------------------------------------------------------------------------


def test_chunk_helper() -> None:
    """_chunk splits list into sublists of the given size."""
    result = _chunk([1, 2, 3, 4, 5], 2)
    assert result == [[1, 2], [3, 4], [5]]
