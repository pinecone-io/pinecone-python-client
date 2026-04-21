"""Evaluation response models for the Assistant API."""

from __future__ import annotations

from typing import Any, Literal

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, safe_display, truncate_text
from pinecone.models.assistant.chat import ChatUsage

EntailmentType = Literal["entailed", "contradicted", "neutral"] | str


class EntailmentResult(Struct, kw_only=True):
    """A single fact with its entailment judgment.

    Attributes:
        fact: The content of the evaluated fact.
        entailment: The entailment classification — one of
            ``"entailed"``, ``"contradicted"``, or ``"neutral"``.
        reasoning: The reasoning behind the entailment judgment.
            Empty string when not provided by the API.
    """

    fact: str
    entailment: EntailmentType
    reasoning: str = ""

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"EntailmentResult(entailment={self.entailment!r},"
            f" fact={truncate_text(self.fact, max_chars=80)!r})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("EntailmentResult(...)")
            return
        with p.group(2, "EntailmentResult(", ")"):
            p.breakable()
            p.text(f"entailment={self.entailment!r},")
            p.breakable()
            p.text(f"fact={truncate_text(self.fact, max_chars=200)!r},")
            if self.reasoning:
                p.breakable()
                p.text(f"reasoning={truncate_text(self.reasoning, max_chars=200)!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("EntailmentResult")
        builder.row("Entailment", self.entailment)
        builder.row("Fact", truncate_text(self.fact, max_chars=500))
        if self.reasoning:
            builder.row("Reasoning", truncate_text(self.reasoning, max_chars=500))
        if self.entailment == "contradicted":
            rows: list[tuple[str, str]] = [
                ("Fact", truncate_text(self.fact, max_chars=500)),
            ]
            if self.reasoning:
                rows.append(("Reasoning", truncate_text(self.reasoning, max_chars=500)))
            builder.section("Contradiction", rows, theme="error")
        return builder.build()


class AlignmentScores(Struct, kw_only=True):
    """Aggregate alignment scores for an evaluation.

    Attributes:
        correctness: Precision of the generated answer.
        completeness: Recall of the generated answer.
        alignment: Harmonic mean of correctness and completeness.
    """

    correctness: float
    completeness: float
    alignment: float


class AlignmentResult(Struct, kw_only=True):
    """Full result of an alignment evaluation.

    Attributes:
        scores: Aggregate correctness, completeness, and alignment scores.
        facts: Per-fact entailment results with reasoning.
        usage: Token usage statistics for the evaluation request.
    """

    scores: AlignmentScores
    facts: list[EntailmentResult]
    usage: ChatUsage
