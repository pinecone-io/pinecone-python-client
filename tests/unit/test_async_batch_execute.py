"""Unit tests for async_batch_execute response_info aggregation."""

from __future__ import annotations

import pytest
from msgspec import Struct

from pinecone._internal.batch import async_batch_execute
from pinecone.models.response_info import ResponseInfo


class _FakeResp(Struct, kw_only=True):
    response_info: ResponseInfo | None = None


def _make_items(n: int) -> list[dict[str, object]]:
    return [{"id": str(i)} for i in range(n)]


class TestAsyncBatchExecuteResponseInfo:
    @pytest.mark.asyncio
    async def test_no_response_info_yields_none(self) -> None:
        async def op(batch: list[dict[str, object]]) -> _FakeResp:
            return _FakeResp()

        result = await async_batch_execute(
            items=_make_items(10),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is None

    @pytest.mark.asyncio
    async def test_aggregates_max_across_successful_batches(self) -> None:
        counter = [0]

        async def op(batch: list[dict[str, object]]) -> _FakeResp:
            counter[0] += 1
            i = counter[0]
            return _FakeResp(
                response_info=ResponseInfo(
                    raw_headers={
                        "x-pinecone-lsn-reconciled": str(i * 10),
                        "x-pinecone-lsn-committed": str(i * 5),
                    }
                )
            )

        result = await async_batch_execute(
            items=_make_items(30),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 30
        assert result.response_info.lsn_committed == 15

    @pytest.mark.asyncio
    async def test_failed_batches_excluded(self) -> None:
        counter = [0]

        async def op(batch: list[dict[str, object]]) -> _FakeResp:
            counter[0] += 1
            if counter[0] == 2:
                raise RuntimeError("middle batch failed")
            return _FakeResp(
                response_info=ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": "50"})
            )

        result = await async_batch_execute(
            items=_make_items(30),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 50

    @pytest.mark.asyncio
    async def test_all_failed_yields_none(self) -> None:
        async def op(batch: list[dict[str, object]]) -> _FakeResp:
            raise RuntimeError("always fails")

        result = await async_batch_execute(
            items=_make_items(20),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is None

    @pytest.mark.asyncio
    async def test_partial_lsn_coverage(self) -> None:
        counter = [0]

        async def op(batch: list[dict[str, object]]) -> _FakeResp:
            counter[0] += 1
            if counter[0] == 1:
                return _FakeResp(
                    response_info=ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": "42"})
                )
            return _FakeResp()

        result = await async_batch_execute(
            items=_make_items(30),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 42

    @pytest.mark.asyncio
    async def test_only_lsn_reconciled_no_committed(self) -> None:
        async def op(batch: list[dict[str, object]]) -> _FakeResp:
            return _FakeResp(
                response_info=ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": "7"})
            )

        result = await async_batch_execute(
            items=_make_items(20),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 7
        assert result.response_info.lsn_committed is None

    @pytest.mark.asyncio
    async def test_operation_returning_none_no_raise(self) -> None:
        async def op(batch: list[dict[str, object]]) -> None:
            return None

        result = await async_batch_execute(
            items=_make_items(10),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is None

    @pytest.mark.asyncio
    async def test_operation_returning_plain_object_no_raise(self) -> None:
        async def op(batch: list[dict[str, object]]) -> object:
            return object()

        result = await async_batch_execute(
            items=_make_items(10),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is None

    @pytest.mark.asyncio
    async def test_malformed_response_info_no_raise(self) -> None:
        class _BadResp:
            response_info = object()

        async def op(batch: list[dict[str, object]]) -> _BadResp:
            return _BadResp()

        result = await async_batch_execute(
            items=_make_items(10),
            operation=op,
            batch_size=10,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is None
