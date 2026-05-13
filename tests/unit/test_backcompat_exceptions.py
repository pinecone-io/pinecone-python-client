"""Backcompat attribute tests for legacy exception aliases."""

from __future__ import annotations

import warnings

import pytest


class TestPineconeApiExceptionStatusAttribute:
    def test_pinecone_api_exception_status_attribute_readable(self) -> None:
        from pinecone.exceptions import PineconeApiException

        exc = PineconeApiException("Something failed", 503)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert exc.status == 503  # type: ignore[attr-defined]
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "status_code" in str(w[0].message)

    def test_pinecone_api_exception_status_code_unchanged(self) -> None:
        from pinecone.exceptions import PineconeApiException

        exc = PineconeApiException("msg", 503)
        assert exc.status_code == 503

    def test_not_found_exception_status_attribute_readable(self) -> None:
        from pinecone.exceptions import NotFoundException

        exc = NotFoundException("not here", 404)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert exc.status == 404  # type: ignore[attr-defined]
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    def test_not_found_exception_status_code_unchanged(self) -> None:
        from pinecone.exceptions import NotFoundException

        exc = NotFoundException("not here", 404)
        assert exc.status_code == 404

    def test_status_deprecation_warning_message(self) -> None:
        from pinecone.exceptions import PineconeApiException

        exc = PineconeApiException("msg", 400)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = exc.status  # type: ignore[attr-defined]
        assert "ApiError.status is deprecated" in str(w[0].message)


class TestPineconeApiTypeErrorPathToItem:
    def test_path_to_item_none_when_no_path(self) -> None:
        from pinecone.exceptions import PineconeApiTypeError

        exc = PineconeApiTypeError("bad type")
        assert exc.path_to_item is None  # type: ignore[attr-defined]

    def test_path_to_item_wraps_path_in_list(self) -> None:
        from pinecone.exceptions import PineconeApiTypeError

        exc = PineconeApiTypeError("bad type", "root.field")
        assert exc.path_to_item == ["root.field"]  # type: ignore[attr-defined]

    def test_path_to_item_is_list_type(self) -> None:
        from pinecone.exceptions import PineconeApiTypeError

        exc = PineconeApiTypeError("bad type", "a.b.c")
        result = exc.path_to_item  # type: ignore[attr-defined]
        assert isinstance(result, list)

    def test_path_attribute_still_accessible(self) -> None:
        from pinecone.exceptions import PineconeApiTypeError

        exc = PineconeApiTypeError("bad type", "root.field")
        assert exc.path == "root.field"


class TestPineconeApiValueErrorPathToItem:
    def test_path_to_item_none_when_no_path(self) -> None:
        from pinecone.exceptions import PineconeApiValueError

        exc = PineconeApiValueError("bad value")
        assert exc.path_to_item is None  # type: ignore[attr-defined]

    def test_path_to_item_wraps_path_in_list(self) -> None:
        from pinecone.exceptions import PineconeApiValueError

        exc = PineconeApiValueError("bad value", "root.field")
        assert exc.path_to_item == ["root.field"]  # type: ignore[attr-defined]

    def test_path_to_item_is_list_type(self) -> None:
        from pinecone.exceptions import PineconeApiValueError

        exc = PineconeApiValueError("bad value", "a.b.c")
        result = exc.path_to_item  # type: ignore[attr-defined]
        assert isinstance(result, list)

    def test_path_attribute_still_accessible(self) -> None:
        from pinecone.exceptions import PineconeApiValueError

        exc = PineconeApiValueError("bad value", "root.field")
        assert exc.path == "root.field"
