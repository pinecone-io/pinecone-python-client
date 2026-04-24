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


def test_sparse_vector_field_custom_metric() -> None:
    schema = SchemaBuilder().add_sparse_vector_field("sparse", metric="custom").build()
    assert schema["fields"]["sparse"]["metric"] == "custom"


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
# add_semantic_text_field
# ---------------------------------------------------------------------------


def test_semantic_text_field_required_params() -> None:
    schema = SchemaBuilder().add_semantic_text_field("text", model="multilingual-e5-large").build()
    field = schema["fields"]["text"]
    assert field["type"] == "semantic_text"
    assert field["model"] == "multilingual-e5-large"
    assert field["metric"] == "cosine"


def test_semantic_text_field_custom_metric() -> None:
    schema = SchemaBuilder().add_semantic_text_field("text", model="m", metric="dotproduct").build()
    assert schema["fields"]["text"]["metric"] == "dotproduct"


def test_semantic_text_field_read_write_parameters() -> None:
    rp: dict[str, Any] = {"input_type": "query"}
    wp: dict[str, Any] = {"input_type": "passage"}
    schema = (
        SchemaBuilder()
        .add_semantic_text_field("text", model="m", read_parameters=rp, write_parameters=wp)
        .build()
    )
    field = schema["fields"]["text"]
    assert field["read_parameters"] == {"input_type": "query"}
    assert field["write_parameters"] == {"input_type": "passage"}


def test_semantic_text_field_omits_none_optional() -> None:
    schema = SchemaBuilder().add_semantic_text_field("text", model="m").build()
    field = schema["fields"]["text"]
    assert "read_parameters" not in field
    assert "write_parameters" not in field
    assert "description" not in field


def test_semantic_text_field_additional_options() -> None:
    schema = SchemaBuilder().add_semantic_text_field("text", model="m", extra="v").build()
    assert schema["fields"]["text"]["extra"] == "v"


# ---------------------------------------------------------------------------
# add_integer_field
# ---------------------------------------------------------------------------


def test_integer_field_defaults() -> None:
    schema = SchemaBuilder().add_integer_field("year").build()
    field = schema["fields"]["year"]
    assert field["type"] == "float"
    assert "filterable" not in field


def test_integer_field_filterable_false_omitted() -> None:
    schema = SchemaBuilder().add_integer_field("year", filterable=False).build()
    assert "filterable" not in schema["fields"]["year"]


def test_integer_field_description() -> None:
    schema = SchemaBuilder().add_integer_field("year", description="pub year").build()
    assert schema["fields"]["year"]["description"] == "pub year"


def test_integer_field_additional_options() -> None:
    schema = SchemaBuilder().add_integer_field("year", extra=1).build()
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
        .add_integer_field("score")
        .add_integer_field("score", filterable=False, description="override")
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
        ("add_semantic_text_field", ("text",), {"model": "multilingual-e5-large"}),
        ("add_integer_field", ("year",), {}),
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
# Re-export from pinecone.preview
# ---------------------------------------------------------------------------


def test_schema_builder_importable_from_preview() -> None:
    from pinecone.preview import PreviewSchemaBuilder

    assert PreviewSchemaBuilder is SchemaBuilder
