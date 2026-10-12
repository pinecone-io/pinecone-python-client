"""SchemaBuilder docs must say what the builder refuses, and no more (#711).

The builder refuses locally only what it can check by type: names, descriptions,
dimension range, and the response-only ``integer`` type. Metadata-only fields
(``float``, ``boolean``, ``string_list``, a ``string`` with no search key) and a
second dense vector field are built without complaint and rejected by the server
on create. The docstrings must not promise otherwise.
"""

from __future__ import annotations

import re

import pytest

from pinecone.models.indexes.schema import IndexSchema
from pinecone.schema_builder import SchemaBuilder


def _flat(doc: str | None) -> str:
    assert doc is not None
    return re.sub(r"\s+", " ", doc)


class TestBuilderBehaviorTheDocsDescribe:
    @pytest.mark.parametrize(
        "add",
        [
            lambda b: b.add_float_field("f"),
            lambda b: b.add_boolean_field("b"),
            lambda b: b.add_string_list_field("l"),
            lambda b: b.add_string_field("s"),
        ],
        ids=["float", "boolean", "string_list", "string_no_search_key"],
    )
    def test_metadata_only_fields_are_built_not_refused(self, add) -> None:  # type: ignore[no-untyped-def]
        assert add(SchemaBuilder()).build()["fields"]

    def test_second_dense_vector_field_is_built_not_refused(self) -> None:
        schema = (
            SchemaBuilder()
            .add_dense_vector_field("a", dimension=2, metric="cosine")
            .add_dense_vector_field("b", dimension=2, metric="cosine")
            .build()
        )
        assert set(schema["fields"]) == {"a", "b"}


class TestDocstringsDoNotOverpromise:
    def test_class_docstring_does_not_claim_server_refusals_are_checked_locally(self) -> None:
        doc = _flat(SchemaBuilder.__doc__)
        assert "the declarations the server refuses are checked" not in doc
        assert "``integer``" in doc
        assert "``400``" in doc

    def test_index_schema_seealso_names_only_integer_as_refused(self) -> None:
        doc = _flat(IndexSchema.__doc__)
        assert "refuses the response-only ones" not in doc
        assert "refuses ``integer`` locally" in doc

    def test_add_dense_vector_field_says_server_enforces_single_field(self) -> None:
        doc = _flat(SchemaBuilder.add_dense_vector_field.__doc__)
        assert re.search(r"at most one.{0,200}server", doc), doc
        assert "does not count" in doc
