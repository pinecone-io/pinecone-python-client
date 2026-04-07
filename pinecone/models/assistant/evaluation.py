"""Evaluation response models for the Assistant API."""

from __future__ import annotations

from typing import Literal

from msgspec import Struct

from pinecone.models.assistant.chat import ChatUsage

EntailmentType = Literal["entailed", "contradicted", "neutral"] | str


class EntailmentResult(Struct, kw_only=True):
    """A single fact with its entailment judgment.

    Attributes:
        fact: The content of the evaluated fact.
        entailment: The entailment classification — one of
            ``"entailed"``, ``"contradicted"``, or ``"neutral"``.
        reasoning: The reasoning behind the entailment judgment.
    """

    fact: str
    entailment: EntailmentType
    reasoning: str


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
