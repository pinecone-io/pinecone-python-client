from __future__ import annotations

import pytest

from pinecone.models.assistant.file_model import AssistantFileModel


def make_full() -> AssistantFileModel:
    return AssistantFileModel(
        name="doc.pdf",
        id="f-1",
        metadata={"k": "v"},
        created_on="2026-01-01T00:00:00Z",
        updated_on="2026-01-02T00:00:00Z",
        status="Available",
        size=1024,
        multimodal=False,
        signed_url="https://example.com/...",
        content_hash="abc123",
        percent_done=100.0,
        error_message=None,
    )


def make_minimal() -> AssistantFileModel:
    return AssistantFileModel(name="a.txt", id="f-x")


def make_failed() -> AssistantFileModel:
    return AssistantFileModel(
        name="a.txt", id="f-x", status="ProcessingFailed", error_message="boom" * 50
    )


class TestRepr:
    def test_populated(self) -> None:
        assert "doc.pdf" in repr(make_full())

    def test_minimal_no_none_noise(self) -> None:
        r = repr(make_minimal())
        assert "None" not in r

    def test_error_message_truncated(self) -> None:
        assert len(repr(make_failed())) < 500

    def test_safe_on_malformed(self) -> None:
        m = make_minimal()
        m.name = object()  # type: ignore[assignment]
        assert isinstance(repr(m), str)


class TestReprHtml:
    def test_populated(self) -> None:
        assert "doc.pdf" in make_full()._repr_html_()

    def test_failed_status_uses_error_section(self) -> None:
        assert "#991b1b" in make_failed()._repr_html_()

    def test_optional_none(self) -> None:
        h = make_minimal()._repr_html_()
        assert "<div" in h

    def test_long_signed_url_truncated(self) -> None:
        m = AssistantFileModel(name="a", id="b", signed_url="https://x/" + "p" * 500)
        assert "..." in m._repr_html_()

    def test_safe_on_malformed(self) -> None:
        m = make_minimal()
        m.id = object()  # type: ignore[assignment]
        assert isinstance(m._repr_html_(), str)


class TestReprPretty:
    def test_populated(self) -> None:
        from IPython.lib.pretty import pretty

        assert "doc.pdf" in pretty(make_full())


@pytest.mark.parametrize("method", ["__repr__", "_repr_html_"])
def test_never_raises(method: str) -> None:
    assert isinstance(getattr(make_full(), method)(), str)
