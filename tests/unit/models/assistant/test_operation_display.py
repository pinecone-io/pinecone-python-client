from __future__ import annotations

import pytest

from pinecone.models.assistant.operation import OperationModel


def make_ok() -> OperationModel:
    return OperationModel(operation_id="op-1", status="Succeeded", created_at="2026-01-01T00:00:00Z")


def make_failed() -> OperationModel:
    return OperationModel(operation_id="op-2", status="Failed", error="boom " * 100)


def make_minimal() -> OperationModel:
    return OperationModel(operation_id="op-3", status="Processing")


class TestRepr:
    def test_ok(self) -> None:
        assert "op-1" in repr(make_ok())

    def test_failed_includes_error(self) -> None:
        r = repr(make_failed())
        assert "Failed" in r
        assert len(r) < 500

    def test_minimal(self) -> None:
        r = repr(make_minimal())
        assert "None" not in r

    def test_safe_on_malformed(self) -> None:
        m = make_ok()
        m.status = object()  # type: ignore[assignment]
        assert isinstance(repr(m), str)


class TestReprHtml:
    def test_ok(self) -> None:
        assert "op-1" in make_ok()._repr_html_()

    def test_failed_has_error_section(self) -> None:
        assert "#991b1b" in make_failed()._repr_html_()

    def test_minimal(self) -> None:
        assert "<div" in make_minimal()._repr_html_()

    def test_safe_on_malformed(self) -> None:
        m = make_ok()
        m.operation_id = object()  # type: ignore[assignment]
        assert isinstance(m._repr_html_(), str)


class TestReprPretty:
    def test_populated(self) -> None:
        from IPython.lib.pretty import pretty

        assert "op-1" in pretty(make_ok())


@pytest.mark.parametrize("method", ["__repr__", "_repr_html_"])
def test_never_raises(method: str) -> None:
    assert isinstance(getattr(make_failed(), method)(), str)
