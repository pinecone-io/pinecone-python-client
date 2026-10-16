"""Halve-and-resend on 413, driven through both engines with a mocked operation."""

from __future__ import annotations

import time
from typing import Any

from pinecone._internal.bulk.async_engine import bulk_execute_async
from pinecone._internal.bulk.engine import bulk_execute_sync
from pinecone.errors.exceptions import ApiError

HOST = "split-test.svc.pinecone.io"
MAX_ITEMS = 8


def _items(n: int) -> list[dict[str, Any]]:
    return [{"_id": str(i)} for i in range(n)]


class _Recorder:
    """Rejects any batch over MAX_ITEMS, or containing the id ``poison``, with 413."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.calls: list[list[str]] = []

    def accept(self, batch: list[dict[str, Any]]) -> dict[str, int]:
        ids = [d["_id"] for d in batch]
        self.calls.append(ids)
        if len(batch) > MAX_ITEMS or "poison" in ids:
            raise ApiError("Payload Too Large", 413)
        self.sent.extend(ids)
        return {"upserted_count": len(batch)}

    async def accept_async(self, batch: list[dict[str, Any]]) -> dict[str, int]:
        return self.accept(batch)


def _sync(rec: _Recorder, items: list[dict[str, Any]], **kw: Any) -> Any:
    return bulk_execute_sync(
        items=items,
        operation=rec.accept,
        batch_size=100,
        show_progress=False,
        host=HOST,
        **kw,
    )


async def _async(rec: _Recorder, items: list[dict[str, Any]], **kw: Any) -> Any:
    return await bulk_execute_async(
        items=items,
        operation=rec.accept_async,
        batch_size=100,
        show_progress=False,
        host=HOST,
        **kw,
    )


def _check_all_sent(rec: _Recorder, result: Any) -> None:
    assert sorted(rec.sent, key=int) == [str(i) for i in range(100)]
    assert result.successful_item_count == 100
    assert result.failed_item_count == 0
    assert result.errors == []
    assert result.successful_batch_count == 1
    assert result.failed_batch_count == 0


def test_sync_splits_until_every_item_is_sent() -> None:
    rec = _Recorder()
    _check_all_sent(rec, _sync(rec, _items(100), split_on_payload_too_large=True))


async def test_async_splits_until_every_item_is_sent() -> None:
    rec = _Recorder()
    _check_all_sent(rec, await _async(rec, _items(100), split_on_payload_too_large=True))


def test_sync_lone_oversize_item_fails_once_and_is_not_retryable() -> None:
    rec = _Recorder()
    items = [*_items(100), {"_id": "poison"}]
    result = _sync(rec, items, split_on_payload_too_large=True)
    assert result.total_item_count == 101
    assert result.failed_item_count == 1
    assert result.successful_item_count == 100
    assert [e.items for e in result.errors] == [[{"_id": "poison"}]]
    assert [e.retryable for e in result.errors] == [False]
    assert rec.calls.count(["poison"]) == 1


async def test_async_lone_oversize_item_fails_once_and_is_not_retryable() -> None:
    rec = _Recorder()
    items = [*_items(100), {"_id": "poison"}]
    result = await _async(rec, items, split_on_payload_too_large=True)
    assert result.failed_item_count == 1
    assert result.successful_item_count == 100
    assert [e.items for e in result.errors] == [[{"_id": "poison"}]]
    assert [e.retryable for e in result.errors] == [False]
    assert rec.calls.count(["poison"]) == 1


def test_sync_without_the_option_a_413_fails_the_whole_batch() -> None:
    rec = _Recorder()
    result = _sync(rec, _items(100))
    assert result.failed_item_count == 100
    assert len(rec.calls) == 1


async def test_async_without_the_option_a_413_fails_the_whole_batch() -> None:
    rec = _Recorder()
    result = await _async(rec, _items(100))
    assert result.failed_item_count == 100
    assert len(rec.calls) == 1


def test_sync_halving_stops_once_the_deadline_has_passed() -> None:
    calls: list[int] = []

    def op(batch: list[dict[str, Any]]) -> Any:
        calls.append(len(batch))
        time.sleep(0.3)
        raise ApiError("Payload Too Large", 413)

    result = bulk_execute_sync(
        items=_items(100),
        operation=op,
        batch_size=100,
        show_progress=False,
        host=HOST,
        split_on_payload_too_large=True,
        total_timeout=0.1,
    )
    assert calls == [100], "pieces were sent after total_timeout expired"
    assert result.timed_out
    assert result.failed_item_count == 100
    assert {e.disposition for e in result.errors} == {"unsent"}
    assert result.failed_batch_count == 1


def test_repr_counts_distinct_batches_not_rows() -> None:
    rec = _Recorder()
    items = [{"_id": "poison"}, {"_id": "poison"}, {"_id": "poison"}, *_items(3)]
    result = _sync(rec, items, split_on_payload_too_large=True)
    assert result.failed_batch_count == 1
    assert "(1 batch)" in repr(result)


def _error_lines(result: Any) -> list[str]:
    return repr(result).split("Errors:\n", 1)[1].splitlines()[:-1]


def test_repr_groups_split_pieces_failing_alike_into_one_line() -> None:
    rec = _Recorder()
    items = [{"_id": "poison"}, {"_id": "poison"}, {"_id": "poison"}, *_items(3)]
    result = _sync(rec, items, split_on_payload_too_large=True)
    assert len(result.errors) == 3
    assert _error_lines(result) == ["    [413] Payload Too Large (1 batch)"]


def test_sync_split_batch_that_landed_in_part_is_a_partial_failure() -> None:
    rec = _Recorder()
    result = _sync(rec, [*_items(3), {"_id": "poison"}], split_on_payload_too_large=True)
    assert result.successful_batch_count == 0
    assert repr(result).startswith("BatchResult(PARTIAL FAILURE: 3/4 items, 0/1 batches")


async def test_async_split_batch_that_landed_in_part_is_a_partial_failure() -> None:
    rec = _Recorder()
    result = await _async(rec, [*_items(3), {"_id": "poison"}], split_on_payload_too_large=True)
    assert result.successful_batch_count == 0
    assert repr(result).startswith("BatchResult(PARTIAL FAILURE: 3/4 items, 0/1 batches")


def test_split_batch_with_every_piece_rejected_is_a_failure() -> None:
    rec = _Recorder()
    result = _sync(rec, [{"_id": "poison"}, {"_id": "poison"}], split_on_payload_too_large=True)
    assert repr(result).startswith("BatchResult(FAILURE: 0/2 items, 0/1 batches")


def _slow_413(batch: list[dict[str, Any]]) -> Any:
    time.sleep(0.3)
    raise ApiError("Payload Too Large", 413)


async def _slow_413_async(batch: list[dict[str, Any]]) -> Any:
    return _slow_413(batch)


def _check_unsent_pieces_grouped(result: Any) -> None:
    assert sorted(len(e.items) for e in result.errors) == [50, 51]
    assert len({e.error_message for e in result.errors}) == 2
    assert len({e.group_key for e in result.errors}) == 1
    assert result.failed_batch_count == 1
    assert len(_error_lines(result)) == 1
    assert _error_lines(result)[0].endswith("(1 batch)")


def test_sync_unsent_split_pieces_of_different_sizes_share_one_repr_line() -> None:
    result = bulk_execute_sync(
        items=_items(101),
        operation=_slow_413,
        batch_size=101,
        show_progress=False,
        host=HOST,
        split_on_payload_too_large=True,
        total_timeout=0.1,
    )
    _check_unsent_pieces_grouped(result)


async def test_async_unsent_split_pieces_of_different_sizes_share_one_repr_line() -> None:
    result = await bulk_execute_async(
        items=_items(101),
        operation=_slow_413_async,
        batch_size=101,
        show_progress=False,
        host=HOST,
        split_on_payload_too_large=True,
        total_timeout=0.1,
    )
    _check_unsent_pieces_grouped(result)
