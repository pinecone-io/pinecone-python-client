"""Unit tests for pinecone._internal.batching helpers."""

from __future__ import annotations

import sys

import pytest

from pinecone._internal.batching import chunked, validate_batch_size, with_progress
from pinecone.errors.exceptions import PineconeValueError


class TestValidateBatchSize:
    def test_validate_batch_size_accepts_positive_int(self) -> None:
        for n in (1, 100, 500, 10_000):
            validate_batch_size(n)

    def test_validate_batch_size_rejects_zero(self) -> None:
        with pytest.raises(PineconeValueError):
            validate_batch_size(0)

    def test_validate_batch_size_rejects_negative(self) -> None:
        with pytest.raises(PineconeValueError):
            validate_batch_size(-1)

    def test_validate_batch_size_rejects_non_int(self) -> None:
        # float and str are rejected
        with pytest.raises(PineconeValueError):
            validate_batch_size(1.5)  # type: ignore[arg-type]
        with pytest.raises(PineconeValueError):
            validate_batch_size("100")  # type: ignore[arg-type]
        with pytest.raises(PineconeValueError):
            validate_batch_size(None)  # type: ignore[arg-type]

    def test_validate_batch_size_accepts_bool(self) -> None:
        # bool is a subclass of int; True == 1 > 0, so it passes (matches
        # the existing upsert_from_dataframe behaviour)
        validate_batch_size(True)  # type: ignore[arg-type]

    def test_validate_batch_size_rejects_false(self) -> None:
        # False == 0, which is not > 0
        with pytest.raises(PineconeValueError):
            validate_batch_size(False)  # type: ignore[arg-type]


class TestChunked:
    def test_chunked_empty_returns_empty_list(self) -> None:
        assert chunked([], 10) == []

    def test_chunked_smaller_than_batch_returns_one_batch(self) -> None:
        assert chunked([1, 2, 3], 10) == [[1, 2, 3]]

    def test_chunked_exact_multiple(self) -> None:
        assert chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_chunked_remainder_in_last_batch(self) -> None:
        assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_chunked_preserves_order_and_identity_for_dicts(self) -> None:
        d1: dict[str, int] = {"a": 1}
        d2: dict[str, int] = {"b": 2}
        d3: dict[str, int] = {"c": 3}
        result = chunked([d1, d2, d3], 2)
        assert result[0][0] is d1
        assert result[0][1] is d2
        assert result[1][0] is d3


class TestWithProgress:
    def test_with_progress_disabled_returns_input(self) -> None:
        items = [1, 2, 3]
        result = with_progress(items, show_progress=False)
        assert list(result) == [1, 2, 3]

    def test_with_progress_no_tqdm_returns_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)  # type: ignore[arg-type]
        items = [1, 2, 3]
        result = with_progress(items, show_progress=True)
        assert list(result) == [1, 2, 3]

    def test_with_progress_with_tqdm_wraps_iterable(self) -> None:
        pytest.importorskip("tqdm")
        items = [1, 2, 3]
        result = with_progress(items, show_progress=True)
        assert result is not items
        assert list(result) == [1, 2, 3]
