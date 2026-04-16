"""Unit tests for preview schema field models."""

from __future__ import annotations

import msgspec
import msgspec.json

from pinecone.preview.models.schema import (
    PreviewDenseVectorField,
    PreviewIntegerField,
    PreviewSchema,
    PreviewSchemaField,
    PreviewSemanticTextField,
    PreviewSparseVectorField,
    PreviewStringField,
)


def test_dense_vector_field_roundtrip() -> None:
    field = PreviewDenseVectorField(dimension=1536, metric="cosine", description="embeddings")
    assert field.dimension == 1536
    assert field.metric == "cosine"
    assert field.description == "embeddings"


def test_sparse_vector_field_defaults() -> None:
    field = PreviewSparseVectorField()
    assert field.metric is None
    assert field.description is None


def test_string_field_fts_and_filterable() -> None:
    field = PreviewStringField(full_text_searchable=True, filterable=True, language="en")
    assert field.full_text_searchable is True
    assert field.filterable is True
    assert field.language == "en"


def test_string_field_stop_words() -> None:
    # Defaults to None
    field_default = PreviewStringField()
    assert field_default.stop_words is None

    # Can be set explicitly
    field_true = PreviewStringField(stop_words=True)
    assert field_true.stop_words is True

    field_false = PreviewStringField(stop_words=False)
    assert field_false.stop_words is False


def test_integer_field_wire_type_is_float() -> None:
    # The msgspec tag for IntegerField is "float" (wire type per API spec)
    raw = b'{"type": "float", "filterable": true}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewIntegerField)
    assert field.filterable is True


def test_schema_field_union_decode() -> None:
    cases: list[tuple[bytes, type]] = [
        (
            b'{"type": "dense_vector", "dimension": 768, "metric": "cosine"}',
            PreviewDenseVectorField,
        ),
        (b'{"type": "sparse_vector", "metric": "dotproduct"}', PreviewSparseVectorField),
        (b'{"type": "semantic_text", "model": "multilingual-e5-large"}', PreviewSemanticTextField),
        (b'{"type": "string", "filterable": true}', PreviewStringField),
        (b'{"type": "float", "filterable": false}', PreviewIntegerField),
    ]
    for raw, expected_type in cases:
        result = msgspec.json.decode(raw, type=PreviewSchemaField)
        assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"


def test_preview_schema_decode() -> None:
    raw = b"""{
        "fields": {
            "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"},
            "title": {"type": "string", "full_text_searchable": true, "language": "en"},
            "year": {"type": "float", "filterable": true},
            "body": {"type": "semantic_text", "model": "multilingual-e5-large"}
        }
    }"""
    schema = msgspec.json.decode(raw, type=PreviewSchema)
    assert isinstance(schema.fields["embedding"], PreviewDenseVectorField)
    assert schema.fields["embedding"].dimension == 1536
    assert isinstance(schema.fields["title"], PreviewStringField)
    assert schema.fields["title"].full_text_searchable is True
    assert isinstance(schema.fields["year"], PreviewIntegerField)
    assert schema.fields["year"].filterable is True
    assert isinstance(schema.fields["body"], PreviewSemanticTextField)


def test_semantic_text_field_with_parameters() -> None:
    field = PreviewSemanticTextField(
        model="multilingual-e5-large",
        metric="cosine",
        read_parameters={"input_type": "query"},
        write_parameters={"input_type": "passage"},
    )
    assert field.model == "multilingual-e5-large"
    assert field.read_parameters == {"input_type": "query"}
    assert field.write_parameters == {"input_type": "passage"}


def test_dense_vector_field_all_defaults() -> None:
    field = PreviewDenseVectorField()
    assert field.dimension is None
    assert field.metric is None
    assert field.description is None


def test_string_field_defaults() -> None:
    field = PreviewStringField()
    assert field.filterable is False
    assert field.full_text_searchable is False
    assert field.language is None
    assert field.stemming is None
    assert field.lowercase is None
    assert field.max_term_len is None
    assert field.stop_words is None


def test_integer_field_defaults() -> None:
    field = PreviewIntegerField()
    assert field.description is None
    assert field.filterable is False


def test_schema_field_union_decode_dense_vector_with_all_fields() -> None:
    raw = (
        b'{"type": "dense_vector", "dimension": 512, "metric": "euclidean", "description": "test"}'
    )
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewDenseVectorField)
    assert field.dimension == 512
    assert field.metric == "euclidean"
    assert field.description == "test"
