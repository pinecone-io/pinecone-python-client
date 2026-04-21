"""Tests for display methods on evaluation models."""

from __future__ import annotations

from pinecone.models.assistant.chat import ChatUsage
from pinecone.models.assistant.evaluation import AlignmentResult, AlignmentScores, EntailmentResult


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


def _usage() -> ChatUsage:
    return ChatUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)


def _mk(n_facts: int, has_contradiction: bool = False) -> AlignmentResult:
    facts = [EntailmentResult(fact=f"fact {i}", entailment="entailed") for i in range(n_facts)]
    if has_contradiction and facts:
        facts[0] = EntailmentResult(fact="bad", entailment="contradicted", reasoning="wrong")
    return AlignmentResult(
        scores=AlignmentScores(correctness=0.9, completeness=0.8, alignment=0.85),
        facts=facts,
        usage=_usage(),
    )


class TestAlignmentResult:
    def test_repr(self) -> None:
        assert "0.85" in repr(_mk(3)) or "0.850" in repr(_mk(3))

    def test_repr_many_facts(self) -> None:
        r = repr(_mk(500))
        assert len(r) < 500

    def test_repr_html(self) -> None:
        assert "<div" in _mk(3)._repr_html_()

    def test_repr_html_contradiction_section(self) -> None:
        h = _mk(3, has_contradiction=True)._repr_html_()
        assert "#991b1b" in h or "contradict" in h.lower()

    def test_repr_html_empty_facts(self) -> None:
        assert "<div" in _mk(0)._repr_html_()

    def test_safe_on_malformed(self) -> None:
        r = _mk(1)
        r.scores = object()  # type: ignore[assignment]
        assert isinstance(repr(r), str)
        assert isinstance(r._repr_html_(), str)
