"""Unit tests for the extended UpsertResponse with BatchResult-superset fields (BC-0106)."""

from __future__ import annotations

from pinecone.models.batch import BatchError, BatchResult
from pinecone.models.vectors.responses import UpsertResponse


def _make_batch_error(batch_index: int, n_items: int) -> BatchError:
    return BatchError(
        batch_index=batch_index,
        items=[{"id": f"v{batch_index}-{i}", "values": [float(i)]} for i in range(n_items)],
        error=RuntimeError(f"batch {batch_index} failed"),
        error_message=f"batch {batch_index} failed",
    )


class TestUpsertResponseDefaultConstruction:
    def test_default_construction_legacy_shape(self) -> None:
        """Non-batched construction preserves all legacy-safe zeros."""
        r = UpsertResponse(upserted_count=10)
        assert r.upserted_count == 10
        assert r.total_item_count == 0
        assert r.failed_item_count == 0
        assert r.total_batch_count == 0
        assert r.successful_batch_count == 0
        assert r.failed_batch_count == 0
        assert r.errors == []
        assert r.has_errors is False
        assert r.error_count == 0
        assert r.success_count == 10
        assert r.successful_item_count == 10
        assert r.failed_items == []

    def test_full_batched_construction(self) -> None:
        """All fields are preserved verbatim when constructed with full batch data."""
        err = _make_batch_error(1, 5)
        r = UpsertResponse(
            upserted_count=200,
            total_item_count=250,
            failed_item_count=50,
            total_batch_count=3,
            successful_batch_count=2,
            failed_batch_count=1,
            errors=[err],
        )
        assert r.upserted_count == 200
        assert r.total_item_count == 250
        assert r.failed_item_count == 50
        assert r.total_batch_count == 3
        assert r.successful_batch_count == 2
        assert r.failed_batch_count == 1
        assert len(r.errors) == 1
        assert r.errors[0] is err
        assert r.has_errors is True
        assert r.error_count == 50
        assert r.success_count == 200
        assert r.successful_item_count == 200


class TestUpsertResponseFailedItems:
    def test_failed_items_flattened_across_batch_errors(self) -> None:
        """failed_items returns all items from all errors in order."""
        err0 = _make_batch_error(0, 5)
        err1 = _make_batch_error(1, 5)
        r = UpsertResponse(upserted_count=0, errors=[err0, err1])
        items = r.failed_items
        assert len(items) == 10
        assert items[:5] == err0.items
        assert items[5:] == err1.items

    def test_failed_items_empty_when_no_errors(self) -> None:
        r = UpsertResponse(upserted_count=100)
        assert r.failed_items == []


class TestUpsertResponseRepr:
    def test_repr_legacy_shape(self) -> None:
        """Non-batched repr is terse legacy style."""
        r = UpsertResponse(upserted_count=10)
        assert repr(r) == "UpsertResponse(upserted_count=10)"

    def test_repr_partial_failure_includes_status(self) -> None:
        """Batched repr with errors shows PARTIAL FAILURE."""
        r = UpsertResponse(
            upserted_count=900,
            total_item_count=1000,
            total_batch_count=10,
            successful_batch_count=9,
            failed_batch_count=1,
            failed_item_count=100,
            errors=[_make_batch_error(2, 100)],
        )
        rep = repr(r)
        assert rep.startswith("UpsertResponse(PARTIAL FAILURE:")
        assert "900/1000 items" in rep
        assert "9/10 batches" in rep

    def test_repr_full_success_batched(self) -> None:
        """Batched repr with no errors shows SUCCESS."""
        r = UpsertResponse(
            upserted_count=1000,
            total_item_count=1000,
            total_batch_count=10,
            successful_batch_count=10,
            failed_batch_count=0,
        )
        rep = repr(r)
        assert rep.startswith("UpsertResponse(SUCCESS:")
        assert "1000/1000 items" in rep
        assert "10/10 batches" in rep


class TestUpsertResponseDictAccess:
    def test_dict_access_through_dict_like_struct(self) -> None:
        """Existing upserted_count key access still works; new fields also accessible."""
        r = UpsertResponse(
            upserted_count=50,
            failed_item_count=10,
            errors=[_make_batch_error(0, 10)],
        )
        assert r["upserted_count"] == 50
        assert r["failed_item_count"] == 10
        assert isinstance(r["errors"], list)
        assert len(r["errors"]) == 1


class TestUpsertResponsePropertyAliasesMatchBatchResult:
    def test_property_aliases_match_batch_result_api(self) -> None:
        """has_errors, error_count, success_count, and failed_items return same values."""
        err = _make_batch_error(0, 5)
        batch_result = BatchResult(
            total_item_count=10,
            successful_item_count=5,
            failed_item_count=5,
            total_batch_count=2,
            successful_batch_count=1,
            failed_batch_count=1,
            errors=[err],
        )
        upsert_response = UpsertResponse(
            upserted_count=5,
            total_item_count=10,
            failed_item_count=5,
            total_batch_count=2,
            successful_batch_count=1,
            failed_batch_count=1,
            errors=[err],
        )
        assert upsert_response.has_errors == batch_result.has_errors
        assert upsert_response.error_count == batch_result.error_count
        assert upsert_response.success_count == batch_result.success_count
        assert upsert_response.failed_items == batch_result.failed_items
