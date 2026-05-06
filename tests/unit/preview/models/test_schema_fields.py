"""Unit tests for preview schema field models."""

from __future__ import annotations

import msgspec
import msgspec.json
import pytest

from pinecone.preview.models.schema import (
    PreviewBooleanField,
    PreviewDenseVectorField,
    PreviewFullTextSearchConfig,
    PreviewIntegerField,
    PreviewLegacyIntegerField,
    PreviewSchema,
    PreviewSchemaField,
    PreviewSemanticTextField,
    PreviewSparseVectorField,
    PreviewStringField,
    PreviewStringListField,
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
    field = PreviewStringField(
        full_text_search=PreviewFullTextSearchConfig(language="en"),
        filterable=True,
    )
    assert field.full_text_search is not None
    assert field.full_text_search.language == "en"
    assert field.filterable is True


def test_string_field_stop_words_config() -> None:
    # Defaults: full_text_search is None (field is not FTS-enabled).
    field_default = PreviewStringField()
    assert field_default.full_text_search is None

    # Empty config: FTS enabled, all options server-defaulted.
    field_empty = PreviewStringField(full_text_search=PreviewFullTextSearchConfig())
    assert field_empty.full_text_search is not None
    assert field_empty.full_text_search.stop_words is None

    # Explicit stop_words on the config.
    field_true = PreviewStringField(full_text_search=PreviewFullTextSearchConfig(stop_words=True))
    assert field_true.full_text_search is not None
    assert field_true.full_text_search.stop_words is True

    field_false = PreviewStringField(full_text_search=PreviewFullTextSearchConfig(stop_words=False))
    assert field_false.full_text_search is not None
    assert field_false.full_text_search.stop_words is False


def test_float_field_wire_type_is_float() -> None:
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
            "title": {"type": "string", "full_text_search": {"language": "en"}},
            "year": {"type": "float", "filterable": true},
            "body": {"type": "semantic_text", "model": "multilingual-e5-large"}
        }
    }"""
    schema = msgspec.json.decode(raw, type=PreviewSchema)
    assert isinstance(schema.fields["embedding"], PreviewDenseVectorField)
    assert schema.fields["embedding"].dimension == 1536
    assert isinstance(schema.fields["title"], PreviewStringField)
    title = schema.fields["title"]
    assert title.full_text_search is not None
    assert title.full_text_search.language == "en"
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
    assert field.full_text_search is None


def test_float_field_model_defaults() -> None:
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


def test_preview_schema_decodes_empty_fields() -> None:
    schema = msgspec.json.decode(b'{"fields": {}}', type=PreviewSchema)
    assert schema.fields == {}
    assert isinstance(schema.fields, dict)


# ---------------------------------------------------------------------------
# PreviewFullTextSearchConfig
# ---------------------------------------------------------------------------


def test_full_text_search_config_all_defaults_none() -> None:
    cfg = PreviewFullTextSearchConfig()
    assert cfg.language is None
    assert cfg.stemming is None
    assert cfg.lowercase is None
    assert cfg.max_term_len is None
    assert cfg.stop_words is None


def test_string_field_decodes_empty_full_text_search_dict() -> None:
    # Per spec: an empty {} config is valid — FTS enabled with all server defaults.
    raw = b'{"type": "string", "full_text_search": {}}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewStringField)
    assert field.full_text_search is not None
    assert field.full_text_search.language is None


def test_string_field_absent_full_text_search_is_none() -> None:
    # Absence = field is not FTS-enabled.
    raw = b'{"type": "string"}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewStringField)
    assert field.full_text_search is None


def test_string_field_decodes_populated_full_text_search() -> None:
    raw = (
        b'{"type": "string", "full_text_search": {"language": "en", '
        b'"stemming": true, "lowercase": false, "max_term_len": 64, "stop_words": true}}'
    )
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewStringField)
    cfg = field.full_text_search
    assert cfg is not None
    assert cfg.language == "en"
    assert cfg.stemming is True
    assert cfg.lowercase is False
    assert cfg.max_term_len == 64
    assert cfg.stop_words is True


# ---------------------------------------------------------------------------
# PreviewStringListField
# ---------------------------------------------------------------------------


def test_string_list_field_defaults() -> None:
    field = PreviewStringListField()
    assert field.filterable is False
    assert field.description is None


def test_string_list_field_with_filterable_and_description() -> None:
    field = PreviewStringListField(filterable=True, description="tags")
    assert field.filterable is True
    assert field.description == "tags"


def test_string_list_field_decode_minimal() -> None:
    raw = b'{"type": "string_list"}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewStringListField)


def test_string_list_field_decode_filterable() -> None:
    raw = b'{"type": "string_list", "filterable": true, "description": "tags"}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewStringListField)
    assert field.filterable is True
    assert field.description == "tags"


def test_schema_field_union_rejects_old_string_array_tag() -> None:
    # The tag "string[]" was renamed to "string_list". Decoding the old tag
    # must now fail at the tagged-union discriminator.
    raw = b'{"type": "string[]"}'
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(raw, type=PreviewSchemaField)


# ---------------------------------------------------------------------------
# PreviewBooleanField
# ---------------------------------------------------------------------------


def test_preview_schema_boolean_field() -> None:
    raw = b'{"type":"boolean","filterable":true}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewBooleanField)
    assert field.filterable is True


def test_preview_schema_boolean_field_defaults() -> None:
    raw = b'{"type":"boolean"}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewBooleanField)
    assert field.filterable is False
    assert field.description is None


def test_preview_schema_boolean_field_with_description() -> None:
    raw = b'{"type":"boolean","filterable":false,"description":"active flag"}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewBooleanField)
    assert field.description == "active flag"


# ---------------------------------------------------------------------------
# PreviewLegacyIntegerField
# ---------------------------------------------------------------------------


def test_preview_schema_integer_field() -> None:
    raw = b'{"type":"integer","filterable":false}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewLegacyIntegerField)
    assert field.filterable is False


def test_preview_schema_integer_field_filterable() -> None:
    raw = b'{"type":"integer","filterable":true}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewLegacyIntegerField)
    assert field.filterable is True


def test_preview_schema_integer_field_defaults() -> None:
    raw = b'{"type":"integer"}'
    field = msgspec.json.decode(raw, type=PreviewSchemaField)
    assert isinstance(field, PreviewLegacyIntegerField)
    assert field.filterable is False
    assert field.description is None


# ---------------------------------------------------------------------------
# Full PreviewIndexModel decode with boolean/integer schema fields
# ---------------------------------------------------------------------------


def test_preview_index_model_with_boolean_field() -> None:
    """Full PreviewIndexModel decode with a boolean schema field must not crash."""
    import msgspec as _msgspec

    from pinecone.preview.models.indexes import PreviewIndexModel

    raw = (
        b'{"name":"idx","host":"idx.svc.pinecone.io",'
        b'"status":{"state":"Ready","ready":true},'
        b'"schema":{"fields":{"is_active":{"type":"boolean","filterable":true}}},'
        b'"deployment":{"deployment_type":"managed","cloud":"aws","region":"us-east-1"},'
        b'"deletion_protection":"disabled"}'
    )
    model = _msgspec.json.decode(raw, type=PreviewIndexModel)
    assert model.name == "idx"
    assert isinstance(model.schema.fields["is_active"], PreviewBooleanField)
    assert model.schema.fields["is_active"].filterable is True


def test_preview_index_model_with_integer_field() -> None:
    """Full PreviewIndexModel decode with a legacy integer schema field must not crash."""
    import msgspec as _msgspec

    from pinecone.preview.models.indexes import PreviewIndexModel

    raw = (
        b'{"name":"idx","host":"idx.svc.pinecone.io",'
        b'"status":{"state":"Ready","ready":true},'
        b'"schema":{"fields":{"count":{"type":"integer","filterable":false}}},'
        b'"deployment":{"deployment_type":"managed","cloud":"aws","region":"us-east-1"},'
        b'"deletion_protection":"disabled"}'
    )
    model = _msgspec.json.decode(raw, type=PreviewIndexModel)
    assert model.name == "idx"
    assert isinstance(model.schema.fields["count"], PreviewLegacyIntegerField)
    assert model.schema.fields["count"].filterable is False
