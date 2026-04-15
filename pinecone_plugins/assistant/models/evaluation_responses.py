"""Backwards-compatibility shim for :mod:`pinecone.models.assistant.evaluation`.

Re-exports evaluation response classes that used to live at
:mod:`pinecone_plugins.assistant.models.evaluation_responses` before the
``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from dataclasses import dataclass
from typing import Any, List

from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass
from pinecone_plugins.assistant.models.shared import TokenCounts


@dataclass
class Fact(BaseDataclass):
    """A single factual claim extracted from a response."""

    content: str

    @classmethod
    def from_openapi(cls, fact: Any) -> "Fact":
        return cls(content=fact.content)


@dataclass
class EvaluatedFact(BaseDataclass):
    """A factual claim with its entailment evaluation result."""

    fact: Fact
    entailment: str

    @classmethod
    def from_openapi(cls, evaluated_fact: Any) -> "EvaluatedFact":
        return cls(
            fact=Fact.from_openapi(evaluated_fact.fact),
            entailment=evaluated_fact.entailment,
        )


@dataclass
class Reasoning(BaseDataclass):
    """Reasoning output containing evaluated facts."""

    evaluated_facts: List[EvaluatedFact]

    @classmethod
    def from_openapi(cls, reasoning: Any) -> "Reasoning":
        return cls(
            evaluated_facts=[EvaluatedFact.from_openapi(f) for f in reasoning.evaluated_facts]
        )


@dataclass
class Metrics(BaseDataclass):
    """Alignment evaluation scores."""

    correctness: float
    completeness: float
    alignment: float

    @classmethod
    def from_openapi(cls, metrics: Any) -> "Metrics":
        return cls(
            correctness=metrics.correctness,
            completeness=metrics.completeness,
            alignment=metrics.alignment,
        )


@dataclass
class AlignmentResponse(BaseDataclass):
    """Full alignment evaluation response."""

    metrics: Metrics
    reasoning: Reasoning
    usage: TokenCounts

    @classmethod
    def from_openapi(cls, alignment_response: Any) -> "AlignmentResponse":
        return cls(
            metrics=Metrics.from_openapi(alignment_response.metrics),
            reasoning=Reasoning.from_openapi(alignment_response.reasoning),
            usage=TokenCounts.from_openapi(alignment_response.usage),
        )


__all__ = [
    "AlignmentResponse",
    "EvaluatedFact",
    "Fact",
    "Metrics",
    "Reasoning",
]
