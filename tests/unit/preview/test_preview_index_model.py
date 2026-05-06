"""Unit tests for PreviewIndexModel optional response fields."""

from __future__ import annotations

import msgspec

from pinecone.preview.models.indexes import PreviewIndexModel

_BASE_RAW = (
    b'{"name":"idx","host":"idx.svc.pinecone.io",'
    b'"status":{"state":"Ready","ready":true},'
    b'"schema":{"fields":{}},'
    b'"deployment":{"deployment_type":"managed","cloud":"aws","region":"us-east-1"},'
    b'"deletion_protection":"disabled"'
)


def test_preview_index_model_optional_response_fields() -> None:
    raw = (
        b'{"name":"idx","host":"idx.svc.pinecone.io",'
        b'"status":{"state":"Ready","ready":true},'
        b'"schema":{"fields":{}},'
        b'"deployment":{"deployment_type":"managed","cloud":"aws","region":"us-east-1"},'
        b'"deletion_protection":"disabled",'
        b'"private_host":"idx.svc.private.pinecone.io",'
        b'"source_collection":"my-collection",'
        b'"source_backup_id":"bkp-abc123",'
        b'"cmek_id":"cmek-xyz789"}'
    )
    model = msgspec.json.decode(raw, type=PreviewIndexModel)
    assert model.private_host == "idx.svc.private.pinecone.io"
    assert model.source_collection == "my-collection"
    assert model.source_backup_id == "bkp-abc123"
    assert model.cmek_id == "cmek-xyz789"


def test_preview_index_model_optional_response_fields_absent() -> None:
    """Fields are None when absent from the response."""
    raw = _BASE_RAW + b"}"
    model = msgspec.json.decode(raw, type=PreviewIndexModel)
    assert model.private_host is None
    assert model.source_collection is None
    assert model.source_backup_id is None
    assert model.cmek_id is None


def test_preview_index_model_repr_includes_private_host_when_set() -> None:
    raw = _BASE_RAW + b',"private_host":"idx.svc.private.pinecone.io"}'
    model = msgspec.json.decode(raw, type=PreviewIndexModel)
    assert "private_host=" in repr(model)
    assert "idx.svc.private.pinecone.io" in repr(model)


def test_preview_index_model_repr_omits_private_host_when_absent() -> None:
    raw = _BASE_RAW + b"}"
    model = msgspec.json.decode(raw, type=PreviewIndexModel)
    assert "private_host" not in repr(model)


def test_preview_index_model_partial_optional_fields() -> None:
    """Only some optional fields present — the rest remain None."""
    raw = (
        b'{"name":"idx","host":"idx.svc.pinecone.io",'
        b'"status":{"state":"Ready","ready":true},'
        b'"schema":{"fields":{}},'
        b'"deployment":{"deployment_type":"managed","cloud":"aws","region":"us-east-1"},'
        b'"deletion_protection":"disabled",'
        b'"source_collection":"col-1"}'
    )
    model = msgspec.json.decode(raw, type=PreviewIndexModel)
    assert model.source_collection == "col-1"
    assert model.private_host is None
    assert model.source_backup_id is None
    assert model.cmek_id is None
