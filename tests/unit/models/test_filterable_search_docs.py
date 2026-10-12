"""The docs must say the server rejects ``filterable`` on a search field with a 400."""

from __future__ import annotations

import inspect
import re

import pytest

from pinecone.models.indexes import schema as schema_module
from pinecone.models.indexes.schema import StringField, _encode_schema_for_request
from pinecone.schema_builder import SchemaBuilder


def _filterable_section(doc: str, marker: str) -> str:
    start = doc.index(marker)
    return re.sub(r"\s+", " ", doc[start : start + 700])


SOURCES = {
    "SchemaBuilder.add_string_field": (
        inspect.getdoc(SchemaBuilder.add_string_field) or "",
        "filterable: Make the field filterable",
    ),
    "StringField.filterable": (inspect.getdoc(StringField) or "", "filterable: Whether the field"),
}


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_filterable_docs_say_server_rejects_with_400(name: str) -> None:
    doc, marker = SOURCES[name]
    text = _filterable_section(doc, marker)
    head = text.split("description:")[0].split("full_text_search:")[0]
    assert "silently" not in head
    assert "discard" not in head
    for term in ("full_text_search", "embed", "sparse_embed", "rejects", "400"):
        assert term in head


def test_encoder_docstring_gives_the_real_reason() -> None:
    doc = re.sub(r"\s+", " ", inspect.getdoc(schema_module._encode_schema_for_request) or "")
    assert "discard" not in doc
    assert "rejects" in doc
    assert "400" in doc


def test_encoder_still_omits_false_filterable_on_search_field() -> None:
    from pinecone.models.indexes.schema import FullTextSearchConfig, IndexSchema

    schema = IndexSchema(fields={"t": StringField(full_text_search=FullTextSearchConfig())})
    assert "filterable" not in _encode_schema_for_request(schema)["fields"]["t"]
    kept = IndexSchema(
        fields={"t": StringField(full_text_search=FullTextSearchConfig(), filterable=True)}
    )
    assert _encode_schema_for_request(kept)["fields"]["t"]["filterable"] is True
