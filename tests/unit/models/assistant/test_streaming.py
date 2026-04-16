"""Unit tests for streaming chunk models — model field presence.

Covers IT-0022: verifies that StreamContentChunk, StreamCitationChunk, and
StreamMessageEnd accept an optional ``model`` field when present in the wire JSON,
and default to ``None`` when absent.

Covers IT-0023: verifies that ChatCompletionStreamChunk accepts the optional
``model``, ``object``, ``created``, and ``system_fingerprint`` fields from the
OpenAI-compatible streaming endpoint.
"""

from __future__ import annotations

import msgspec

from pinecone.models.assistant.streaming import (
    ChatCompletionStreamChunk,
    ChatStreamChunk,
    StreamCitationChunk,
    StreamContentChunk,
    StreamMessageEnd,
    StreamMessageStart,
)

# ---------------------------------------------------------------------------
# StreamContentChunk — model field
# ---------------------------------------------------------------------------


def test_stream_content_chunk_accepts_model() -> None:
    """StreamContentChunk decodes a JSON fixture that includes a model field."""
    raw = b'{"type": "content_chunk", "id": "c1", "delta": {"content": "Hello"}, "model": "gpt-4o"}'
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamContentChunk)
    assert chunk.id == "c1"
    assert chunk.delta.content == "Hello"
    assert chunk.model == "gpt-4o"


def test_stream_content_chunk_model_defaults_none() -> None:
    """StreamContentChunk.model defaults to None when absent from the wire response."""
    raw = b'{"type": "content_chunk", "id": "c1", "delta": {"content": "Hello"}}'
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamContentChunk)
    assert chunk.model is None


# ---------------------------------------------------------------------------
# StreamCitationChunk — model field
# ---------------------------------------------------------------------------


def test_stream_citation_chunk_accepts_model() -> None:
    """StreamCitationChunk decodes a JSON fixture that includes a model field."""
    raw = (
        b'{"type": "citation", "id": "cit1",'
        b' "citation": {"position": 5, "references": []},'
        b' "model": "gpt-4o"}'
    )
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamCitationChunk)
    assert chunk.id == "cit1"
    assert chunk.model == "gpt-4o"


def test_stream_citation_chunk_model_defaults_none() -> None:
    """StreamCitationChunk.model defaults to None when absent from the wire response."""
    raw = b'{"type": "citation", "id": "cit1", "citation": {"position": 5, "references": []}}'
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamCitationChunk)
    assert chunk.model is None


# ---------------------------------------------------------------------------
# StreamMessageEnd — model field
# ---------------------------------------------------------------------------


def test_stream_message_end_accepts_model() -> None:
    """StreamMessageEnd decodes a JSON fixture that includes a model field."""
    raw = (
        b'{"type": "message_end", "id": "end1",'
        b' "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},'
        b' "model": "gpt-4o"}'
    )
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamMessageEnd)
    assert chunk.id == "end1"
    assert chunk.usage.total_tokens == 15
    assert chunk.model == "gpt-4o"


def test_stream_message_end_model_defaults_none() -> None:
    """StreamMessageEnd.model defaults to None when absent from the wire response."""
    raw = (
        b'{"type": "message_end", "id": "end1",'
        b' "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}'
    )
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamMessageEnd)
    assert chunk.model is None


# ---------------------------------------------------------------------------
# StreamMessageStart — model field (already present, sanity check)
# ---------------------------------------------------------------------------


def test_stream_message_start_has_model() -> None:
    """StreamMessageStart.model was already required; confirm it still works."""
    raw = b'{"type": "message_start", "model": "gpt-4o", "role": "assistant"}'
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamMessageStart)
    assert chunk.model == "gpt-4o"
    assert chunk.role == "assistant"


# ---------------------------------------------------------------------------
# ChatCompletionStreamChunk — extra fields (IT-0023)
# ---------------------------------------------------------------------------


def test_chat_completion_stream_chunk_accepts_extra_fields() -> None:
    """ChatCompletionStreamChunk decodes a realistic JSON fixture with all OpenAI fields."""
    raw = (
        b'{"id": "chatcmpl-abc123",'
        b' "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null}],'
        b' "model": "gpt-4o",'
        b' "object": "chat.completion.chunk",'
        b' "created": 1714000000,'
        b' "system_fingerprint": "fp_abc123"}'
    )
    chunk = msgspec.json.decode(raw, type=ChatCompletionStreamChunk)
    assert chunk.id == "chatcmpl-abc123"
    assert len(chunk.choices) == 1
    assert chunk.choices[0].index == 0
    assert chunk.choices[0].delta.content == "Hello"
    assert chunk.model == "gpt-4o"
    assert chunk.object == "chat.completion.chunk"
    assert chunk.created == 1714000000
    assert chunk.system_fingerprint == "fp_abc123"


def test_chat_completion_stream_chunk_extra_fields_default_none() -> None:
    """ChatCompletionStreamChunk optional fields default to None when absent."""
    raw = b'{"id": "chatcmpl-xyz", "choices": []}'
    chunk = msgspec.json.decode(raw, type=ChatCompletionStreamChunk)
    assert chunk.id == "chatcmpl-xyz"
    assert chunk.choices == []
    assert chunk.model is None
    assert chunk.object is None
    assert chunk.created is None
    assert chunk.system_fingerprint is None


def test_chat_completion_stream_chunk_model_only() -> None:
    """ChatCompletionStreamChunk with only model field set (no object/created/fingerprint)."""
    raw = (
        b'{"id": "chatcmpl-model-only",'
        b' "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}],'
        b' "model": "gpt-4o-mini"}'
    )
    chunk = msgspec.json.decode(raw, type=ChatCompletionStreamChunk)
    assert chunk.id == "chatcmpl-model-only"
    assert chunk.model == "gpt-4o-mini"
    assert chunk.object is None
    assert chunk.created is None
    assert chunk.system_fingerprint is None
