"""Tests for display methods on evaluation models."""
from __future__ import annotations

from pinecone.models.assistant.evaluation import EntailmentResult


class TestEntailmentResult:
    def test_repr(self) -> None:
        r = repr(EntailmentResult(fact="the sky is blue", entailment="entailed"))
        assert "entailed" in r and "sky" in r

    def test_repr_long_fact_truncated(self) -> None:
        r = repr(EntailmentResult(fact="x" * 5000, entailment="neutral"))
        assert len(r) < 500

    def test_repr_html_contradicted_shows_error_theme(self) -> None:
        h = EntailmentResult(fact="f", entailment="contradicted", reasoning="bad")._repr_html_()
        assert "#991b1b" in h or "contradicted" in h

    def test_repr_html_without_reasoning(self) -> None:
        h = EntailmentResult(fact="f", entailment="entailed")._repr_html_()
        assert "<div" in h

    def test_safe_on_malformed(self) -> None:
        e = EntailmentResult(fact="x", entailment="entailed")
        e.entailment = object()  # type: ignore[assignment]
        assert isinstance(repr(e), str)
        assert isinstance(e._repr_html_(), str)
