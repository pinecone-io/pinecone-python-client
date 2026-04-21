from __future__ import annotations

import pytest

from pinecone.models.assistant.message import Message


def make_short() -> Message:
    return Message(content="Hello", role="user")


def make_long() -> Message:
    return Message(content="x" * 10_000, role="assistant")


class TestRepr:
    def test_short(self) -> None:
        r = repr(make_short())
        assert "Hello" in r
        assert "user" in r

    def test_long_truncated(self) -> None:
        r = repr(make_long())
        assert len(r) < 500
        assert "..." in r

    def test_safe_on_malformed(self) -> None:
        m = make_short()
        m.content = object()  # type: ignore[assignment]
        assert isinstance(repr(m), str)


class TestReprHtml:
    def test_short(self) -> None:
        assert "Hello" in make_short()._repr_html_()

    def test_long_truncated(self) -> None:
        h = make_long()._repr_html_()
        assert len(h) < 5000
        assert "..." in h

    def test_safe_on_malformed(self) -> None:
        m = make_short()
        m.role = object()  # type: ignore[assignment]
        assert isinstance(m._repr_html_(), str)


class TestReprPretty:
    def test_populated(self) -> None:
        from IPython.lib.pretty import pretty

        assert "Hello" in pretty(make_short())


@pytest.mark.parametrize("method", ["__repr__", "_repr_html_"])
def test_never_raises(method: str) -> None:
    assert isinstance(getattr(make_long(), method)(), str)
