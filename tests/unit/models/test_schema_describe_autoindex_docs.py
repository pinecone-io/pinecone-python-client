"""The schema docs must not promise that describe reports auto-indexed metadata."""

from __future__ import annotations

import inspect
import re

import msgspec.json
import pytest

from pinecone.models.indexes.schema import (
    BooleanField,
    FloatField,
    IndexSchema,
    StringField,
    StringListField,
)

_METADATA_FIELDS = [FloatField, BooleanField, StringListField]


def _flat(obj: object) -> str:
    return re.sub(r"\s+", " ", inspect.getdoc(obj) or "")


@pytest.mark.parametrize("cls", _METADATA_FIELDS)
def test_metadata_field_docs_say_describe_lists_only_declared_fields(cls: type) -> None:
    doc = _flat(cls)
    assert "``describe`` lists only the fields you declared" in doc
    assert "the server indexed for you at upsert time" not in doc


def test_index_schema_docs_say_describe_lists_only_declared_fields() -> None:
    doc = _flat(IndexSchema)
    assert "``describe`` on a documents index lists only the fields you declared" in doc
    assert "indexed for filtering automatically at upsert time" not in doc
    assert "things you read back" not in doc


def test_describe_of_a_documents_index_decodes_to_only_the_declared_fields() -> None:
    raw = b'{"fields": {"f_true": {"type": "string", "full_text_search": {}}}}'
    schema = msgspec.json.decode(raw, type=IndexSchema)
    assert list(schema.fields) == ["f_true"]
    assert isinstance(schema.fields["f_true"], StringField)
