"""Tests for display methods on evaluation models."""
from __future__ import annotations

from pinecone.models.assistant.evaluation import AlignmentScores, EntailmentResult


class TestEntailmentResult:
    def test_repr(self) -> None:
        r = repr(EntailmentResult(fact="the sky is blue", entailment="entailed"))
        assert "entailed" in r
        assert "sky" in r

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


class TestAlignmentScores:
    def test_repr(self) -> None:
        r = repr(AlignmentScores(correctness=0.8, completeness=0.9, alignment=0.85))
        assert "0.8" in r or "0.800" in r

    def test_repr_html(self) -> None:
        assert "<div" in AlignmentScores(correctness=1, completeness=1, alignment=1)._repr_html_()

    def test_safe_on_malformed(self) -> None:
        s = AlignmentScores(correctness=0, completeness=0, alignment=0)
        s.correctness = object()  # type: ignore[assignment]
        assert isinstance(repr(s), str)
        assert isinstance(s._repr_html_(), str)
