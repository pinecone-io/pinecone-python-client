"""IndexModel's deprecated vector accessors treat a string field's ``embed`` as a dense field."""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from pinecone.models.indexes.index import IndexModel


def _model(fields: dict[str, Any]) -> IndexModel:
    return msgspec.convert(
        {
            "name": "arxiv-docs-10k-embed",
            "host": "h",
            "deletion_protection": "disabled",
            "status": {"ready": True, "state": "Ready"},
            "deployment": {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
            "schema": {"fields": fields},
        },
        IndexModel,
    )


_EMBED_TITLE = {
    "type": "string",
    "full_text_search": {"language": "en"},
    "embed": {"model": "llama-text-embed-v2", "dimension": 1024, "metric": "cosine"},
}


def test_string_embed_field_resolves_dimension_metric_and_vector_type() -> None:
    model = _model({"title": _EMBED_TITLE})
    assert model.dimension == 1024
    assert model.metric == "cosine"
    assert model.vector_type == "dense"


def test_string_embed_field_resolves_through_item_access_and_to_dict() -> None:
    model = _model({"title": _EMBED_TITLE})
    assert model["dimension"] == 1024
    assert ("metric" in model) is hasattr(model, "metric")
    assert model["metric"] == model.metric
    assert model.to_dict()["vector_type"] == "dense"


def test_string_embed_with_sparse_embed_still_resolves_dense() -> None:
    field = {**_EMBED_TITLE, "sparse_embed": {}}
    model = _model({"title": field})
    assert (model.dimension, model.metric, model.vector_type) == (1024, "cosine", "dense")


@pytest.mark.parametrize("accessor", ["dimension", "metric", "vector_type"])
def test_string_embed_alongside_dense_vector_field_is_ambiguous(accessor: str) -> None:
    model = _model(
        {
            "title": _EMBED_TITLE,
            "vec": {"type": "dense_vector", "dimension": 8, "metric": "dotproduct"},
        }
    )
    with pytest.raises(AttributeError, match="ambiguous") as exc:
        getattr(model, accessor)
    assert "title" in str(exc.value)
    assert "vec" in str(exc.value)


def test_two_string_embed_fields_are_ambiguous() -> None:
    model = _model({"a": _EMBED_TITLE, "b": _EMBED_TITLE})
    with pytest.raises(AttributeError, match="ambiguous"):
        model.dimension


def test_string_field_without_embed_still_has_no_vector() -> None:
    model = _model({"title": {"type": "string", "full_text_search": {}}})
    with pytest.raises(AttributeError, match="could not be determined"):
        model.dimension


def test_embed_without_resolved_dimension_or_metric_raises() -> None:
    model = _model({"title": {"type": "string", "embed": {}}})
    with pytest.raises(AttributeError):
        model.dimension
    with pytest.raises(AttributeError):
        model.metric
    assert model.vector_type == "dense"


def test_embed_without_resolved_value_names_the_field() -> None:
    model = _model({"title": {"type": "string", "embed": {}}})
    with pytest.raises(AttributeError, match="'title' does not report a dimension"):
        model.dimension


@pytest.mark.parametrize("accessor", ["dimension", "metric"])
def test_embed_without_resolved_value_points_at_embed_attribute(accessor: str) -> None:
    model = _model({"title": {"type": "string", "embed": {}}})
    with pytest.raises(AttributeError) as exc:
        getattr(model, accessor)
    assert "index.schema.fields['title'].embed" in str(exc.value)
    assert not hasattr(model.schema.fields["title"], accessor)
    assert hasattr(model.schema.fields["title"].embed, accessor)


def test_ambiguity_message_marks_string_embed_fields() -> None:
    model = _model(
        {
            "title": _EMBED_TITLE,
            "vec": {"type": "dense_vector", "dimension": 8, "metric": "dotproduct"},
        }
    )
    with pytest.raises(AttributeError, match=r"title \(string embed\)"):
        model.dimension


def test_string_embed_with_sparse_vector_field_resolves_dense() -> None:
    model = _model({"title": _EMBED_TITLE, "sp": {"type": "sparse_vector"}})
    assert (model.dimension, model.metric, model.vector_type) == (1024, "cosine", "dense")
