from __future__ import annotations

import pytest

from pinecone.models.assistant.options import ContextOptions


def make_full() -> ContextOptions:
    return ContextOptions(top_k=10, snippet_size=512, multimodal=True, include_binary_content=False)


def make_empty() -> ContextOptions:
    return ContextOptions()


class TestRepr:
    def test_full(self) -> None:
        r = repr(make_full())
        assert "top_k=10" in r

    def test_all_none(self) -> None:
        r = repr(make_empty())
        assert "default" in r.lower() or r == "ContextOptions()"

    def test_safe_on_malformed(self) -> None:
        m = make_full()
        m.top_k = object()  # type: ignore[assignment]
        assert isinstance(repr(m), str)


class TestReprHtml:
    def test_full(self) -> None:
        h = make_full()._repr_html_()
        assert "top_k" in h.lower() or "Top" in h

    def test_all_none(self) -> None:
        h = make_empty()._repr_html_()
        assert "<div" in h

    def test_safe_on_malformed(self) -> None:
        m = make_full()
        m.snippet_size = object()  # type: ignore[assignment]
        assert isinstance(m._repr_html_(), str)


class TestReprPretty:
    def test_populated(self) -> None:
        from IPython.lib.pretty import pretty

        assert "top_k" in pretty(make_full())


@pytest.mark.parametrize("method", ["__repr__", "_repr_html_"])
def test_never_raises(method: str) -> None:
    assert isinstance(getattr(make_full(), method)(), str)
    assert isinstance(getattr(make_empty(), method)(), str)
