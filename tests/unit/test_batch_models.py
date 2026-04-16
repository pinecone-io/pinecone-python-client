"""Tests for BatchError and BatchResult models."""

from __future__ import annotations

import json

from pinecone.models.batch import BatchError, BatchResult


def make_batch_error(
    batch_index: int = 0,
    items: list[dict] | None = None,
    error: Exception | None = None,
    error_message: str = "Connection timeout",
) -> BatchError:
    return BatchError(
        batch_index=batch_index,
        items=items if items is not None else [{"id": "v1", "values": [0.1, 0.2]}],
        error=error if error is not None else RuntimeError("Connection timeout"),
        error_message=error_message,
    )


def make_batch_result(
    total_item_count: int = 100,
    successful_item_count: int = 100,
    failed_item_count: int = 0,
    total_batch_count: int = 10,
    successful_batch_count: int = 10,
    failed_batch_count: int = 0,
    errors: list[BatchError] | None = None,
) -> BatchResult:
    return BatchResult(
        total_item_count=total_item_count,
        successful_item_count=successful_item_count,
        failed_item_count=failed_item_count,
        total_batch_count=total_batch_count,
        successful_batch_count=successful_batch_count,
        failed_batch_count=failed_batch_count,
        errors=errors if errors is not None else [],
    )


class TestBatchError:
    def test_batch_error_repr(self) -> None:
        err = make_batch_error(
            batch_index=3,
            items=[{"id": "a"}, {"id": "b"}],
            error=ValueError("bad request"),
            error_message="bad request",
        )
        r = repr(err)
        assert "3" in r
        assert "2" in r  # item_count
        assert "bad request" in r

    def test_batch_error_to_dict(self) -> None:
        exc = RuntimeError("timeout")
        err = make_batch_error(
            batch_index=1,
            items=[{"id": "x"}],
            error=exc,
            error_message="timeout",
        )
        d = err.to_dict()
        assert d["batch_index"] == 1
        assert d["items"] == [{"id": "x"}]
        assert isinstance(d["error"], str)
        assert d["error"] == str(exc)
        assert d["error_message"] == "timeout"

    def test_batch_error_to_dict_error_is_not_exception(self) -> None:
        exc = ValueError("invalid vector")
        err = make_batch_error(error=exc, error_message="invalid vector")
        d = err.to_dict()
        assert not isinstance(d["error"], Exception)
        assert isinstance(d["error"], str)

    def test_batch_error_to_json(self) -> None:
        exc = RuntimeError("oops")
        err = make_batch_error(
            batch_index=2,
            items=[{"id": "y"}],
            error=exc,
            error_message="oops",
        )
        result = err.to_json()
        parsed = json.loads(result)
        assert parsed["batch_index"] == 2
        assert isinstance(parsed["error"], str)
        assert parsed["error_message"] == "oops"


class TestBatchResultSuccess:
    def test_batch_result_success_has_errors_false(self) -> None:
        result = make_batch_result()
        assert result.has_errors is False

    def test_batch_result_success_failed_items_empty(self) -> None:
        result = make_batch_result()
        assert result.failed_items == []

    def test_batch_result_success_repr_contains_success(self) -> None:
        result = make_batch_result(
            total_item_count=50,
            successful_item_count=50,
            total_batch_count=5,
            successful_batch_count=5,
        )
        r = repr(result)
        assert "SUCCESS" in r
        assert "50/50" in r
        assert "5/5" in r

    def test_batch_result_success_repr_no_errors_section(self) -> None:
        result = make_batch_result()
        r = repr(result)
        assert "PARTIAL FAILURE" not in r
        assert "Errors" not in r


class TestBatchResultPartialFailure:
    def test_batch_result_partial_failure_has_errors_true(self) -> None:
        err = make_batch_error()
        result = make_batch_result(
            successful_item_count=90,
            failed_item_count=10,
            successful_batch_count=9,
            failed_batch_count=1,
            errors=[err],
        )
        assert result.has_errors is True

    def test_batch_result_partial_failure_failed_items_flattened(self) -> None:
        items_a = [{"id": "a1"}, {"id": "a2"}]
        items_b = [{"id": "b1"}]
        err_a = make_batch_error(batch_index=0, items=items_a)
        err_b = make_batch_error(batch_index=1, items=items_b)
        result = make_batch_result(
            failed_item_count=3,
            failed_batch_count=2,
            errors=[err_a, err_b],
        )
        assert result.failed_items == items_a + items_b

    def test_batch_result_partial_failure_repr_contains_partial_failure(self) -> None:
        err = make_batch_error(error_message="upstream error")
        result = make_batch_result(
            successful_item_count=90,
            failed_item_count=10,
            successful_batch_count=9,
            failed_batch_count=1,
            errors=[err],
        )
        r = repr(result)
        assert "PARTIAL FAILURE" in r
        assert "upstream error" in r

    def test_batch_result_partial_failure_repr_no_success(self) -> None:
        err = make_batch_error()
        result = make_batch_result(errors=[err])
        r = repr(result)
        assert "SUCCESS" not in r


class TestBatchResultReprHtml:
    def test_batch_result_repr_html_success(self) -> None:
        result = make_batch_result()
        html = result._repr_html_()
        assert "<div" in html
        assert "BatchResult" in html
        assert "Total items:" in html
        # No error section
        assert "#fdf2f2" not in html
        assert "#991b1b" not in html

    def test_batch_result_repr_html_with_errors(self) -> None:
        err = make_batch_error(error_message="rate limit exceeded")
        result = make_batch_result(
            failed_item_count=10,
            failed_batch_count=1,
            errors=[err],
        )
        html = result._repr_html_()
        assert "<div" in html
        assert "BatchResult" in html
        # Summary table present
        assert "Total items:" in html
        # Error section present (red-themed)
        assert "#fdf2f2" in html
        assert "#991b1b" in html
        assert "rate limit exceeded" in html

    def test_batch_result_repr_html_escapes_error_messages(self) -> None:
        err = make_batch_error(error_message="<script>alert(1)</script>")
        result = make_batch_result(errors=[err])
        html = result._repr_html_()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestBatchResultToDict:
    def test_batch_result_to_dict_all_fields(self) -> None:
        result = make_batch_result(
            total_item_count=100,
            successful_item_count=90,
            failed_item_count=10,
            total_batch_count=10,
            successful_batch_count=9,
            failed_batch_count=1,
        )
        d = result.to_dict()
        assert d["total_item_count"] == 100
        assert d["successful_item_count"] == 90
        assert d["failed_item_count"] == 10
        assert d["total_batch_count"] == 10
        assert d["successful_batch_count"] == 9
        assert d["failed_batch_count"] == 1
        assert d["errors"] == []

    def test_batch_result_to_dict_errors_have_string_exceptions(self) -> None:
        exc = RuntimeError("disk full")
        err = make_batch_error(error=exc, error_message="disk full")
        result = make_batch_result(errors=[err])
        d = result.to_dict()
        assert len(d["errors"]) == 1
        assert isinstance(d["errors"][0]["error"], str)
        assert d["errors"][0]["error"] == str(exc)

    def test_batch_result_to_json(self) -> None:
        err = make_batch_error(batch_index=0, error_message="bad gateway")
        result = make_batch_result(errors=[err])
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["total_item_count"] == 100
        assert len(parsed["errors"]) == 1
        assert isinstance(parsed["errors"][0]["error"], str)


class TestBatchResultErrorSummary:
    def test_batch_result_error_summary_dedup(self) -> None:
        errors = [make_batch_error(batch_index=i, error_message="timeout") for i in range(3)] + [
            make_batch_error(batch_index=3, error_message="rate limit"),
        ]
        result = make_batch_result(errors=errors)
        summary = result._error_summary()
        # Should be sorted by frequency (most common first)
        assert summary[0] == ("timeout", 3)
        assert summary[1] == ("rate limit", 1)

    def test_batch_result_error_summary_empty(self) -> None:
        result = make_batch_result()
        assert result._error_summary() == []

    def test_batch_result_repr_plural_batch_word(self) -> None:
        errors = [make_batch_error(batch_index=i, error_message="conn refused") for i in range(2)]
        result = make_batch_result(errors=errors)
        r = repr(result)
        assert "batches" in r

    def test_batch_result_repr_singular_batch_word(self) -> None:
        err = make_batch_error(error_message="unique error")
        result = make_batch_result(errors=[err])
        r = repr(result)
        assert "1 batch)" in r
