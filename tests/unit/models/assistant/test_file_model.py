"""Unit tests for AssistantFileModel wire-format and backwards-compatibility."""

from __future__ import annotations

import msgspec

from pinecone.models.assistant.file_model import AssistantFileModel


def test_crc32c_hash_alias() -> None:
    """crc32c_hash property returns the same value as content_hash."""
    model = AssistantFileModel(name="f.txt", id="id-1", content_hash="abc123")
    assert model.content_hash == "abc123"
    assert model.crc32c_hash == "abc123"


def test_crc32c_hash_alias_none() -> None:
    """crc32c_hash property returns None when content_hash is None."""
    model = AssistantFileModel(name="f.txt", id="id-1")
    assert model.content_hash is None
    assert model.crc32c_hash is None


def test_wire_key_is_crc32c_hash() -> None:
    """msgspec deserializes the wire key 'crc32c_hash' into the content_hash attribute."""
    raw = b'{"name":"f.txt","id":"id-1","crc32c_hash":"deadbeef"}'
    model = msgspec.json.decode(raw, type=AssistantFileModel)
    assert model.content_hash == "deadbeef"
    assert model.crc32c_hash == "deadbeef"


def test_wire_key_content_hash_ignored() -> None:
    """msgspec silently ignores an unknown 'content_hash' wire key (not the struct key)."""
    raw = b'{"name":"f.txt","id":"id-1","content_hash":"should-be-ignored"}'
    model = msgspec.json.decode(raw, type=AssistantFileModel)
    # content_hash in JSON is not the wire key — it maps to nothing;
    # the Python attribute content_hash should remain None.
    assert model.content_hash is None
    assert model.crc32c_hash is None


def test_optional_fields_default_none() -> None:
    """All optional fields default to None when absent from the wire response."""
    raw = b'{"name":"f.txt","id":"id-1"}'
    model = msgspec.json.decode(raw, type=AssistantFileModel)
    assert model.metadata is None
    assert model.created_on is None
    assert model.updated_on is None
    assert model.status is None
    assert model.size is None
    assert model.multimodal is None
    assert model.signed_url is None
    assert model.content_hash is None
    assert model.crc32c_hash is None
    assert model.percent_done is None
    assert model.error_message is None
