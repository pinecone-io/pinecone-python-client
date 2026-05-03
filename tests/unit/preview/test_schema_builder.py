"""Unit tests for PreviewSchemaBuilder."""

from __future__ import annotations

from typing import Any

import pytest

from pinecone.preview import PreviewSchemaBuilder as SchemaBuilder

# ---------------------------------------------------------------------------
# add_dense_vector_field
# ---------------------------------------------------------------------------


def test_dense_vector_field_basic() -> None:
    schema = SchemaBuilder().add_dense_vector_field("vec", dimension=768, metric="cosine").build()
    assert schema == {
        "fields": {"vec": {"type": "dense_vector", "dimension": 768, "metric": "cosine"}}
    }


def test_dense_vector_field_with_description() -> None:
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("vec", dimension=1536, metric="dotproduct", description="ada-002")
        .build()
    )
    assert schema["fields"]["vec"]["description"] == "ada-002"


def test_dense_vector_field_additional_options() -> None:
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("vec", dimension=64, metric="cosine", extra="val")
        .build()
    )
    assert schema["fields"]["vec"]["extra"] == "val"


def test_dense_vector_field_no_description_omitted() -> None:
    schema = SchemaBuilder().add_dense_vector_field("vec", dimension=64, metric="cosine").build()
    assert "description" not in schema["fields"]["vec"]


# ---------------------------------------------------------------------------
# add_sparse_vector_field
# ---------------------------------------------------------------------------


def test_sparse_vector_field_defaults() -> None:
    schema = SchemaBuilder().add_sparse_vector_field("sparse").build()
    field = schema["fields"]["sparse"]
    assert field["type"] == "sparse_vector"
    assert field["metric"] == "dotproduct"
    assert "description" not in field


def test_sparse_vector_field_metric_always_dotproduct() -> None:
    schema = SchemaBuilder().add_sparse_vector_field("sparse").build()
    assert schema["fields"]["sparse"]["metric"] == "dotproduct"


def test_sparse_vector_field_with_description() -> None:
    schema = SchemaBuilder().add_sparse_vector_field("sparse", description="BM25").build()
    assert schema["fields"]["sparse"]["description"] == "BM25"


def test_sparse_vector_field_additional_options() -> None:
    schema = SchemaBuilder().add_sparse_vector_field("sparse", extra=True).build()
    assert schema["fields"]["sparse"]["extra"] is True


# ---------------------------------------------------------------------------
# add_string_field
# ---------------------------------------------------------------------------


def test_string_field_defaults_omit_false_booleans() -> None:
    schema = SchemaBuilder().add_string_field("title").build()
    field = schema["fields"]["title"]
    assert field["type"] == "string"
    assert "full_text_search" not in field
    assert "filterable" not in field


def test_string_field_full_text_search_empty_dict() -> None:
    # Empty dict is valid — signals FTS-enabled with server defaults for all options.
    schema = SchemaBuilder().add_string_field("title", full_text_search={}).build()
    assert schema["fields"]["title"]["full_text_search"] == {}


def test_string_field_full_text_search_with_language() -> None:
    schema = SchemaBuilder().add_string_field("title", full_text_search={"language": "en"}).build()
    assert schema["fields"]["title"]["full_text_search"] == {"language": "en"}


def test_string_field_filterable_true() -> None:
    schema = SchemaBuilder().add_string_field("cat", filterable=True).build()
    assert schema["fields"]["cat"]["filterable"] is True


def test_string_field_omits_full_text_search_when_not_provided() -> None:
    schema = SchemaBuilder().add_string_field("t").build()
    field = schema["fields"]["t"]
    assert "full_text_search" not in field
    # No flat FTS option keys leak to the top level.
    for key in ("language", "stemming", "lowercase", "max_term_len", "stop_words"):
        assert key not in field


def test_string_field_full_text_search_all_options() -> None:
    # lowercase and max_term_len are server-managed but still round-trip through dict.
    cfg = {
        "language": "en",
        "stemming": True,
        "lowercase": False,
        "max_term_len": 40,
        "stop_words": False,
    }
    schema = (
        SchemaBuilder()
        .add_string_field("body", full_text_search=cfg, description="article body")
        .build()
    )
    field = schema["fields"]["body"]
    assert field["full_text_search"] == cfg
    assert field["description"] == "article body"


def test_string_field_full_text_search_true_emits_empty_dict() -> None:
    schema = SchemaBuilder().add_string_field("t", full_text_search=True).build()
    assert schema["fields"]["t"]["full_text_search"] == {}


def test_string_field_kwargs_imply_fts_enabled() -> None:
    schema = SchemaBuilder().add_string_field("t", language="en").build()
    assert schema["fields"]["t"]["full_text_search"] == {"language": "en"}


def test_string_field_kwarg_normalizes_long_alias() -> None:
    schema = SchemaBuilder().add_string_field("t", language="english").build()
    assert schema["fields"]["t"]["full_text_search"] == {"language": "en"}


def test_string_field_kwarg_normalizes_long_alias_in_dict() -> None:
    schema = SchemaBuilder().add_string_field("t", full_text_search={"language": "english"}).build()
    assert schema["fields"]["t"]["full_text_search"] == {"language": "en"}


def test_string_field_kwargs_merge_into_dict() -> None:
    schema = (
        SchemaBuilder()
        .add_string_field("t", full_text_search={"language": "fr"}, stemming=True)
        .build()
    )
    fts = schema["fields"]["t"]["full_text_search"]
    assert fts["language"] == "fr"
    assert fts["stemming"] is True


def test_string_field_kwarg_overrides_dict_for_same_key() -> None:
    schema = (
        SchemaBuilder()
        .add_string_field("t", full_text_search={"language": "fr"}, language="en")
        .build()
    )
    assert schema["fields"]["t"]["full_text_search"]["language"] == "en"


def test_string_field_unknown_language_passes_through_kwarg() -> None:
    schema = SchemaBuilder().add_string_field("t", language="klingon").build()
    assert schema["fields"]["t"]["full_text_search"] == {"language": "klingon"}


def test_string_field_unknown_language_passes_through_dict() -> None:
    schema = (
        SchemaBuilder().add_string_field("t", full_text_search={"language": "klingon"}).build()
    )
    assert schema["fields"]["t"]["full_text_search"] == {"language": "klingon"}


def test_string_field_stop_words_without_stemming_raises() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="stop_words requires stemming to be enabled"):
        SchemaBuilder().add_string_field("t", language="en", stop_words=True)


def test_string_field_stop_words_with_stemming_passes() -> None:
    schema = (
        SchemaBuilder().add_string_field("t", language="en", stemming=True, stop_words=True).build()
    )
    fts = schema["fields"]["t"]["full_text_search"]
    assert fts["language"] == "en"
    assert fts["stemming"] is True
    assert fts["stop_words"] is True


def test_string_field_stop_words_via_dict_validates() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="stop_words requires stemming to be enabled"):
        SchemaBuilder().add_string_field("t", full_text_search={"stop_words": True})


def test_string_field_does_not_validate_language_stop_words_compat() -> None:
    # Arabic does not support stop words, but rule 3 is server-side only.
    schema = (
        SchemaBuilder().add_string_field("t", language="ar", stemming=True, stop_words=True).build()
    )
    fts = schema["fields"]["t"]["full_text_search"]
    assert fts["language"] == "ar"
    assert fts["stop_words"] is True


def test_string_field_lowercase_and_max_term_len_pass_through() -> None:
    cfg = {"lowercase": False, "max_term_len": 40}
    schema = SchemaBuilder().add_string_field("t", full_text_search=cfg).build()
    fts = schema["fields"]["t"]["full_text_search"]
    assert fts["lowercase"] is False
    assert fts["max_term_len"] == 40


def test_string_field_full_text_search_dict_is_copied_not_aliased() -> None:
    cfg = {"language": "en"}
    builder = SchemaBuilder().add_string_field("title", full_text_search=cfg)
    cfg["language"] = "fr"  # mutate after the call
    assert builder.build()["fields"]["title"]["full_text_search"] == {"language": "en"}


def test_string_field_additional_options_merged() -> None:
    schema = SchemaBuilder().add_string_field("t", future_param="x").build()
    assert schema["fields"]["t"]["future_param"] == "x"


def test_string_field_additional_options_merged_last() -> None:
    # additional_options override explicit kwargs because they .update() last.
    schema = SchemaBuilder().add_string_field("t", extra_future_key="x").build()
    assert schema["fields"]["t"]["extra_future_key"] == "x"
    assert schema["fields"]["t"]["type"] == "string"


def test_string_field_full_text_and_filterable_together() -> None:
    schema = (
        SchemaBuilder()
        .add_string_field(
            "title",
            full_text_search={"language": "en"},
            filterable=True,
        )
        .build()
    )
    field = schema["fields"]["title"]
    assert field["full_text_search"] == {"language": "en"}
    assert field["filterable"] is True


# ---------------------------------------------------------------------------
# add_float_field
# ---------------------------------------------------------------------------


def test_float_field_defaults() -> None:
    schema = SchemaBuilder().add_float_field("year").build()
    field = schema["fields"]["year"]
    assert field["type"] == "float"
    assert "filterable" not in field


def test_float_field_filterable_false_omitted() -> None:
    schema = SchemaBuilder().add_float_field("year", filterable=False).build()
    assert "filterable" not in schema["fields"]["year"]


def test_float_field_description() -> None:
    schema = SchemaBuilder().add_float_field("year", description="pub year").build()
    assert schema["fields"]["year"]["description"] == "pub year"


def test_float_field_additional_options() -> None:
    schema = SchemaBuilder().add_float_field("year", extra=1).build()
    assert schema["fields"]["year"]["extra"] == 1


# ---------------------------------------------------------------------------
# add_custom_field
# ---------------------------------------------------------------------------


def test_custom_field_stored_verbatim() -> None:
    raw: dict[str, Any] = {"type": "new_type", "foo": 42}
    schema = SchemaBuilder().add_custom_field("experimental", raw).build()
    assert schema["fields"]["experimental"] == {"type": "new_type", "foo": 42}


def test_custom_field_complex_definition() -> None:
    raw: dict[str, Any] = {"type": "new_type", "nested": {"a": 1}, "list_val": [1, 2]}
    schema = SchemaBuilder().add_custom_field("f", raw).build()
    assert schema["fields"]["f"]["type"] == "new_type"
    assert schema["fields"]["f"]["nested"] == {"a": 1}
    assert schema["fields"]["f"]["list_val"] == [1, 2]


# ---------------------------------------------------------------------------
# Re-adding a field name replaces it
# ---------------------------------------------------------------------------


def test_duplicate_field_name_replaces() -> None:
    schema = (
        SchemaBuilder()
        .add_string_field("title")
        .add_dense_vector_field("title", dimension=64, metric="cosine")
        .build()
    )
    assert schema["fields"]["title"]["type"] == "dense_vector"
    assert len(schema["fields"]) == 1


def test_duplicate_field_preserves_last_definition() -> None:
    schema = (
        SchemaBuilder()
        .add_float_field("score")
        .add_float_field("score", filterable=False, description="override")
        .build()
    )
    assert schema["fields"]["score"]["description"] == "override"
    assert "filterable" not in schema["fields"]["score"]


# ---------------------------------------------------------------------------
# add_* methods return self for chaining
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    [
        ("add_dense_vector_field", ("vec",), {"dimension": 128, "metric": "cosine"}),
        ("add_sparse_vector_field", ("sparse",), {}),
        ("add_string_field", ("title",), {}),
        ("add_string_list_field", ("tags",), {}),
        ("add_float_field", ("year",), {}),
        ("add_boolean_field", ("is_published",), {}),
        ("add_custom_field", ("custom", {"type": "custom"}), {}),
    ],
)
def test_add_methods_return_self_for_chaining(
    method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    builder = SchemaBuilder()
    method = getattr(builder, method_name)
    assert method(*args, **kwargs) is builder


# ---------------------------------------------------------------------------
# build() idempotency
# ---------------------------------------------------------------------------


def test_build_idempotent_same_content() -> None:
    builder = SchemaBuilder().add_string_field("title")
    result1 = builder.build()
    result2 = builder.build()
    assert result1 == result2


def test_build_returns_copy_not_same_object() -> None:
    builder = SchemaBuilder().add_string_field("title")
    result1 = builder.build()
    result2 = builder.build()
    assert result1 is not result2
    assert result1["fields"] is not result2["fields"]


def test_build_subsequent_mutations_do_not_affect_prior_result() -> None:
    builder = SchemaBuilder().add_string_field("title")
    first = builder.build()
    builder.add_string_field("body")
    second = builder.build()
    assert "body" not in first["fields"]
    assert "body" in second["fields"]


def test_build_empty_schema_returns_empty_fields() -> None:
    result = SchemaBuilder().build()
    assert result == {"fields": {}}
    assert isinstance(result["fields"], dict)


# ---------------------------------------------------------------------------
# add_string_list_field
# ---------------------------------------------------------------------------


def test_string_list_field_defaults() -> None:
    schema = SchemaBuilder().add_string_list_field("tags").build()
    field = schema["fields"]["tags"]
    assert field == {"type": "string_list"}


def test_string_list_field_filterable_true() -> None:
    schema = SchemaBuilder().add_string_list_field("tags", filterable=True).build()
    assert schema["fields"]["tags"]["filterable"] is True


def test_string_list_field_description() -> None:
    schema = (
        SchemaBuilder().add_string_list_field("tags", description="genres and keywords").build()
    )
    assert schema["fields"]["tags"]["description"] == "genres and keywords"


def test_string_list_field_additional_options_merged() -> None:
    schema = SchemaBuilder().add_string_list_field("tags", future_key="v").build()
    assert schema["fields"]["tags"]["future_key"] == "v"


def test_string_list_field_emits_snake_case_tag_not_brackets() -> None:
    # Sanity-check: the old "string[]" tag must never appear.
    schema = SchemaBuilder().add_string_list_field("tags").build()
    assert schema["fields"]["tags"]["type"] == "string_list"
    assert schema["fields"]["tags"]["type"] != "string[]"


# ---------------------------------------------------------------------------
# add_boolean_field
# ---------------------------------------------------------------------------


def test_boolean_field_defaults() -> None:
    schema = SchemaBuilder().add_boolean_field("is_published").build()
    field = schema["fields"]["is_published"]
    assert field == {"type": "boolean"}


def test_boolean_field_filterable_true() -> None:
    schema = SchemaBuilder().add_boolean_field("is_published", filterable=True).build()
    assert schema["fields"]["is_published"]["filterable"] is True


def test_boolean_field_filterable_false_omitted() -> None:
    schema = SchemaBuilder().add_boolean_field("is_published", filterable=False).build()
    assert "filterable" not in schema["fields"]["is_published"]


def test_boolean_field_description() -> None:
    schema = (
        SchemaBuilder().add_boolean_field("is_published", description="visibility flag").build()
    )
    assert schema["fields"]["is_published"]["description"] == "visibility flag"


def test_boolean_field_additional_options_merged() -> None:
    schema = SchemaBuilder().add_boolean_field("is_published", future_key="v").build()
    assert schema["fields"]["is_published"]["future_key"] == "v"


# ---------------------------------------------------------------------------
# Field-name and description validation
# ---------------------------------------------------------------------------


def test_field_name_empty_raises() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="non-empty"):
        SchemaBuilder().add_string_field("")


def test_field_name_starts_with_dollar_raises() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match=r"\$.*_"):
        SchemaBuilder().add_string_field("$illegal")


def test_field_name_starts_with_underscore_raises() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match=r"\$.*_"):
        SchemaBuilder().add_string_field("_illegal")


def test_field_name_over_64_bytes_raises() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    long_name = "a" * 65
    with pytest.raises(PineconeValueError, match="too long"):
        SchemaBuilder().add_string_field(long_name)


def test_field_name_64_bytes_ok() -> None:
    schema = SchemaBuilder().add_string_field("a" * 64).build()
    assert "a" * 64 in schema["fields"]


def test_field_name_multibyte_counts_bytes_not_chars() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    multibyte_name = "é" * 33  # "é" is 2 bytes in UTF-8 -> 66 bytes total
    with pytest.raises(PineconeValueError, match="too long"):
        SchemaBuilder().add_string_field(multibyte_name)


def test_description_over_256_bytes_raises() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="too long"):
        SchemaBuilder().add_string_field("title", description="x" * 257)


def test_description_256_bytes_ok() -> None:
    schema = SchemaBuilder().add_string_field("title", description="x" * 256).build()
    assert schema["fields"]["title"]["description"] == "x" * 256


def test_description_none_ok() -> None:
    schema = SchemaBuilder().add_string_field("title").build()
    assert "description" not in schema["fields"]["title"]


def test_validation_applies_to_every_add_method() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    builders = [
        lambda: SchemaBuilder().add_dense_vector_field("$bad", dimension=4, metric="cosine"),
        lambda: SchemaBuilder().add_sparse_vector_field("$bad"),
        lambda: SchemaBuilder().add_string_field("$bad"),
        lambda: SchemaBuilder().add_string_list_field("$bad"),
        lambda: SchemaBuilder().add_boolean_field("$bad"),
        lambda: SchemaBuilder().add_float_field("$bad"),
        lambda: SchemaBuilder().add_custom_field("$bad", {"type": "x"}),
    ]
    for build in builders:
        with pytest.raises(PineconeValueError):
            build()


# ---------------------------------------------------------------------------
# Re-export from pinecone.preview
# ---------------------------------------------------------------------------


def test_schema_builder_importable_from_preview() -> None:
    from pinecone.preview import PreviewSchemaBuilder

    assert PreviewSchemaBuilder is SchemaBuilder
