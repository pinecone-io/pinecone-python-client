"""Unit tests for streaming chunk models — model field presence.

Covers IT-0022: verifies that StreamContentChunk, StreamCitationChunk, and
StreamMessageEnd accept an optional ``model`` field when present in the wire JSON,
and default to ``None`` when absent.
"""

from __future__ import annotations

import msgspec

from pinecone.models.assistant.streaming import (
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
