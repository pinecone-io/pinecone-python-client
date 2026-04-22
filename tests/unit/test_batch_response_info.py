"""Tests for BatchResponseInfo and the response_info field on BatchResult."""

from __future__ import annotations

from pinecone.models.batch import BatchResult
from pinecone.models.response_info import BatchResponseInfo


def _make_batch_result(**kwargs: object) -> BatchResult:
    defaults: dict[str, object] = {
        "total_item_count": 10,
        "successful_item_count": 10,
        "failed_item_count": 0,
        "total_batch_count": 1,
        "successful_batch_count": 1,
        "failed_batch_count": 0,
        "errors": [],
    }
    defaults.update(kwargs)
    return BatchResult(**defaults)  # type: ignore[arg-type]


class TestBatchResponseInfo:
    def test_default_values_all_none(self) -> None:
        info = BatchResponseInfo()
        assert info.lsn_reconciled is None
        assert info.lsn_committed is None

    def test_kwargs_construct(self) -> None:
        info = BatchResponseInfo(lsn_reconciled=42, lsn_committed=40)
        assert info.lsn_reconciled == 42
        assert info.lsn_committed == 40

    def test_is_reconciled_true_equal(self) -> None:
        assert BatchResponseInfo(lsn_reconciled=10).is_reconciled(10) is True

    def test_is_reconciled_true_exceeds(self) -> None:
        assert BatchResponseInfo(lsn_reconciled=10).is_reconciled(5) is True

    def test_is_reconciled_false_below(self) -> None:
        assert BatchResponseInfo(lsn_reconciled=5).is_reconciled(10) is False

    def test_is_reconciled_false_when_none(self) -> None:
        assert BatchResponseInfo().is_reconciled(1) is False

    def test_no_raw_headers_attr(self) -> None:
        info = BatchResponseInfo()
        assert not hasattr(info, "raw_headers")

    def test_no_request_id_attr(self) -> None:
        info = BatchResponseInfo()
        assert not hasattr(info, "request_id")

    def test_to_dict(self) -> None:
        result = BatchResponseInfo(lsn_reconciled=7, lsn_committed=9).to_dict()
        assert result == {"lsn_reconciled": 7, "lsn_committed": 9}


class TestBatchResultResponseInfo:
    def test_batch_result_response_info_defaults_to_none(self) -> None:
        result = _make_batch_result()
        assert result.response_info is None

    def test_batch_result_accepts_response_info(self) -> None:
        result = _make_batch_result(response_info=BatchResponseInfo(lsn_reconciled=42))
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 42

    def test_batch_result_to_dict_includes_response_info_none(self) -> None:
        result = _make_batch_result()
        assert result.to_dict()["response_info"] is None

    def test_batch_result_to_dict_includes_response_info_populated(self) -> None:
        result = _make_batch_result(
            response_info=BatchResponseInfo(lsn_reconciled=7, lsn_committed=9)
        )
        assert result.to_dict()["response_info"] == {"lsn_reconciled": 7, "lsn_committed": 9}


class TestImportable:
    def test_importable_from_pinecone(self) -> None:
        from pinecone import BatchResponseInfo

        assert BatchResponseInfo is not None
