"""Adapter for Assistants API responses."""

from __future__ import annotations

import orjson

from pinecone._internal.adapters._decode import decode_response
from pinecone.errors.exceptions import ResponseParsingError
from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse, ChatUsage
from pinecone.models.assistant.context import ContextResponse
from pinecone.models.assistant.evaluation import AlignmentResult, AlignmentScores, EntailmentResult
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
from pinecone.models.assistant.model import AssistantModel


class AssistantsAdapter:
    """Transforms raw API JSON into AssistantModel / ListAssistantsResponse instances."""

    @staticmethod
    def to_assistant(data: bytes) -> AssistantModel:
        """Decode raw JSON bytes into an AssistantModel."""
        return decode_response(data, AssistantModel)

    @staticmethod
    def to_assistant_list(data: bytes) -> ListAssistantsResponse:
        """Decode raw JSON bytes into a ListAssistantsResponse."""
        return decode_response(data, ListAssistantsResponse)

    @staticmethod
    def to_file(data: bytes) -> AssistantFileModel:
        """Decode raw JSON bytes into an AssistantFileModel."""
        return decode_response(data, AssistantFileModel)

    @staticmethod
    def to_file_list(data: bytes) -> ListFilesResponse:
        """Decode raw JSON bytes into a ListFilesResponse."""
        return decode_response(data, ListFilesResponse)

    @staticmethod
    def to_chat_response(data: bytes) -> ChatResponse:
        """Decode raw JSON bytes into a ChatResponse."""
        return decode_response(data, ChatResponse)

    @staticmethod
    def to_chat_completion_response(data: bytes) -> ChatCompletionResponse:
        """Decode raw JSON bytes into a ChatCompletionResponse."""
        return decode_response(data, ChatCompletionResponse)

    @staticmethod
    def to_context_response(data: bytes) -> ContextResponse:
        """Decode raw JSON bytes into a ContextResponse."""
        return decode_response(data, ContextResponse)

    @staticmethod
    def to_alignment_result(data: bytes) -> AlignmentResult:
        """Decode raw JSON bytes into an AlignmentResult.

        Transforms the API response shape (``metrics``, ``reasoning``, ``usage``)
        into the SDK model shape (``scores``, ``facts``, ``usage``).
        """
        try:
            raw: dict[str, object] = orjson.loads(data)
            metrics = raw["metrics"]
            assert isinstance(metrics, dict)
            scores = AlignmentScores(
                correctness=float(metrics["correctness"]),
                completeness=float(metrics["completeness"]),
                alignment=float(metrics["alignment"]),
            )
            reasoning = raw["reasoning"]
            assert isinstance(reasoning, dict)
            evaluated_facts = reasoning["evaluated_facts"]
            assert isinstance(evaluated_facts, list)
            facts = [
                EntailmentResult(
                    fact=str(item["fact"]["content"]),
                    entailment=str(item["entailment"]),
                )
                for item in evaluated_facts
            ]
            u = raw["usage"]
            assert isinstance(u, dict)
            usage = ChatUsage(
                prompt_tokens=int(u["prompt_tokens"]),
                completion_tokens=int(u["completion_tokens"]),
                total_tokens=int(u["total_tokens"]),
            )
            return AlignmentResult(scores=scores, facts=facts, usage=usage)
        except (KeyError, TypeError, ValueError, AssertionError) as exc:
            raise ResponseParsingError(
                f"Failed to parse API response as AlignmentResult: {exc}",
                cause=exc,
            ) from exc
