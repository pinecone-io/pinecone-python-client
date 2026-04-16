"""Unit tests for the evaluation.metrics.alignment() legacy proxy on AsyncAssistants."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pinecone.async_client.assistants import AsyncAssistants
from pinecone.models.assistant.chat import ChatUsage
from pinecone.models.assistant.evaluation import AlignmentResult, AlignmentScores, EntailmentResult

_CANNED_ALIGNMENT = AlignmentResult(
    scores=AlignmentScores(correctness=1.0, completeness=1.0, alignment=1.0),
    facts=[EntailmentResult(fact="2024", entailment="entailed", reasoning="matches")],
    usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
)


@pytest.fixture
def mock_evaluate_alignment(mock_async_assistants: AsyncAssistants) -> AsyncMock:
    """Stub AsyncAssistants.evaluate_alignment to return a canned AlignmentResult."""
    spy = AsyncMock(return_value=_CANNED_ALIGNMENT)
    mock_async_assistants.evaluate_alignment = spy  # type: ignore[method-assign]
    return spy


async def test_async_evaluation_metrics_alignment_legacy(
    mock_async_assistants: AsyncAssistants, mock_evaluate_alignment: AsyncMock
) -> None:
    result = await mock_async_assistants.evaluation.metrics.alignment(
        question="What year?",
        answer="2024",
        ground_truth_answer="2024",
    )
    assert hasattr(result, "scores") or isinstance(result, AlignmentResult)


async def test_async_evaluation_metrics_alignment_delegates(
    mock_async_assistants: AsyncAssistants, mock_evaluate_alignment: AsyncMock
) -> None:
    """alignment() forwards arguments to evaluate_alignment()."""
    await mock_async_assistants.evaluation.metrics.alignment(
        question="Q",
        answer="A",
        ground_truth_answer="GT",
    )
    mock_evaluate_alignment.assert_called_once_with(
        question="Q",
        answer="A",
        ground_truth_answer="GT",
    )


async def test_async_evaluation_proxy_cached(mock_async_assistants: AsyncAssistants) -> None:
    p1 = mock_async_assistants.evaluation
    p2 = mock_async_assistants.evaluation
    assert p1 is p2


async def test_async_evaluation_metrics_proxy_cached(
    mock_async_assistants: AsyncAssistants,
) -> None:
    m1 = mock_async_assistants.evaluation.metrics
    m2 = mock_async_assistants.evaluation.metrics
    assert m1 is m2


async def test_async_evaluation_metrics_returns_alignment_result(
    mock_async_assistants: AsyncAssistants, mock_evaluate_alignment: AsyncMock
) -> None:
    result = await mock_async_assistants.evaluation.metrics.alignment(
        question="What year?",
        answer="2024",
        ground_truth_answer="2024",
    )
    assert isinstance(result, AlignmentResult)
    assert result.scores.alignment == 1.0
