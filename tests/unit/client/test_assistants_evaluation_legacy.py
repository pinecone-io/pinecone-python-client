"""Unit tests for the evaluation.metrics.alignment() legacy proxy on Assistants."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone.client.assistants import Assistants
from pinecone.models.assistant.chat import ChatUsage
from pinecone.models.assistant.evaluation import AlignmentResult, AlignmentScores, EntailmentResult

_CANNED_ALIGNMENT = AlignmentResult(
    scores=AlignmentScores(correctness=1.0, completeness=1.0, alignment=1.0),
    facts=[EntailmentResult(fact="2024", entailment="entailed", reasoning="matches")],
    usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
)


@pytest.fixture
def mock_evaluate_alignment(mock_assistants: Assistants) -> MagicMock:
    """Stub Assistants.evaluate_alignment to return a canned AlignmentResult."""
    spy = MagicMock(return_value=_CANNED_ALIGNMENT)
    mock_assistants.evaluate_alignment = spy  # type: ignore[method-assign]
    return spy


def test_evaluation_metrics_alignment_legacy(
    mock_assistants: Assistants, mock_evaluate_alignment: MagicMock
) -> None:
    result = mock_assistants.evaluation.metrics.alignment(
        question="What year?",
        answer="2024",
        ground_truth_answer="2024",
    )
    assert hasattr(result, "scores") or isinstance(result, AlignmentResult)


def test_evaluation_metrics_alignment_delegates(
    mock_assistants: Assistants, mock_evaluate_alignment: MagicMock
) -> None:
    """alignment() forwards arguments to evaluate_alignment()."""
    mock_assistants.evaluation.metrics.alignment(
        question="Q",
        answer="A",
        ground_truth_answer="GT",
    )
    mock_evaluate_alignment.assert_called_once_with(
        question="Q",
        answer="A",
        ground_truth_answer="GT",
    )


def test_evaluation_proxy_cached(mock_assistants: Assistants) -> None:
    p1 = mock_assistants.evaluation
    p2 = mock_assistants.evaluation
    assert p1 is p2


def test_evaluation_metrics_proxy_cached(mock_assistants: Assistants) -> None:
    m1 = mock_assistants.evaluation.metrics
    m2 = mock_assistants.evaluation.metrics
    assert m1 is m2


def test_evaluation_metrics_returns_alignment_result(
    mock_assistants: Assistants, mock_evaluate_alignment: MagicMock
) -> None:
    result = mock_assistants.evaluation.metrics.alignment(
        question="What year?",
        answer="2024",
        ground_truth_answer="2024",
    )
    assert isinstance(result, AlignmentResult)
    assert result.scores.alignment == 1.0
