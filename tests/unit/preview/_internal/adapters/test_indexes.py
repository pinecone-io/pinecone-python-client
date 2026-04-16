"""Unit tests for preview index adapters."""

from __future__ import annotations

import orjson

from pinecone.preview._internal.adapters.indexes import (
    _filter_none,
    configure_adapter,
    create_adapter,
    describe_adapter,
    list_adapter,
)
from pinecone.preview.models.deployment import PreviewManagedDeployment
from pinecone.preview.models.indexes import PreviewIndexModel
from pinecone.preview.models.requests import PreviewConfigureIndexRequest, PreviewCreateIndexRequest


def _minimal_index_dict(name: str = "test-idx") -> dict:  # type: ignore[type-arg]
    return {
        "name": name,
        "host": f"{name}-host.pinecone.io",
        "status": {"ready": True, "state": "Ready"},
        "schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}},
        "deployment": {
            "deployment_type": "managed",
            "environment": "us-east-1-aws",
            "cloud": "aws",
            "region": "us-east-1",
        },
        "deletion_protection": "disabled",
    }


# ---------------------------------------------------------------------------
# create_adapter.to_request
# ---------------------------------------------------------------------------


def test_create_adapter_serializes_minimal() -> None:
    req = PreviewCreateIndexRequest(
        schema={"fields": {"e": {"type": "dense_vector", "dimension": 4}}}
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    assert set(data.keys()) == {"schema"}
    assert data["schema"]["fields"]["e"]["type"] == "dense_vector"


def test_create_adapter_preserves_explicit_false_overrides() -> None:
    req = PreviewCreateIndexRequest(
        schema={
            "fields": {
                "title": {"type": "string", "filterable": False, "full_text_searchable": True}
            }
        }
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    field = data["schema"]["fields"]["title"]
    assert field.get("filterable") is False
    assert field.get("full_text_searchable") is True


def test_create_adapter_preserves_user_explicit_lowercase_false() -> None:
    req = PreviewCreateIndexRequest(
        schema={
            "fields": {
                "title": {"type": "string", "full_text_searchable": True, "lowercase": False}
            }
        }
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    field = data["schema"]["fields"]["title"]
    assert field.get("lowercase") is False
    assert field.get("full_text_searchable") is True


def test_create_adapter_preserves_stemming_false_and_stop_words_false() -> None:
    req = PreviewCreateIndexRequest(
        schema={
            "fields": {
                "title": {
                    "type": "string",
                    "full_text_searchable": True,
                    "stemming": False,
                    "stop_words": False,
                }
            }
        }
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    field = data["schema"]["fields"]["title"]
    assert field.get("stemming") is False
    assert field.get("stop_words") is False
    assert field.get("full_text_searchable") is True


def test_create_adapter_serializes_full_request() -> None:
    req = PreviewCreateIndexRequest(
        schema={"fields": {"e": {"type": "dense_vector", "dimension": 4}}},
        name="my-idx",
        deployment={
            "deployment_type": "managed",
            "environment": "us-east-1-aws",
            "cloud": "aws",
            "region": "us-east-1",
        },
        read_capacity={"mode": "OnDemand"},
        tags={"env": "prod"},
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    assert data["name"] == "my-idx"
    assert data["tags"] == {"env": "prod"}
    assert data["deployment"]["deployment_type"] == "managed"


def test_create_adapter_preserves_unknown_additional_options() -> None:
    req = PreviewCreateIndexRequest(
        schema={
            "fields": {
                "embedding": {
                    "type": "dense_vector",
                    "dimension": 768,
                    "metric": "cosine",
                    "new_future_param": "value",
                }
            }
        }
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    field = data["schema"]["fields"]["embedding"]
    assert field["new_future_param"] == "value"
    assert field["type"] == "dense_vector"
    assert field["dimension"] == 768


def test_create_adapter_preserves_add_custom_field_unknown_type() -> None:
    req = PreviewCreateIndexRequest(
        schema={"fields": {"experimental": {"type": "new_type", "foo": 42}}}
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    assert data["schema"]["fields"]["experimental"] == {"type": "new_type", "foo": 42}


# ---------------------------------------------------------------------------
# create_adapter.from_response
# ---------------------------------------------------------------------------


def test_create_adapter_parses_response() -> None:
    data = _minimal_index_dict("new-idx")
    model = create_adapter.from_response(data)
    assert isinstance(model, PreviewIndexModel)
    assert model.name == "new-idx"
    assert model.host == "new-idx-host.pinecone.io"
    assert model.status.ready is True
    assert isinstance(model.deployment, PreviewManagedDeployment)


def test_create_adapter_parses_response_with_tags() -> None:
    data = {**_minimal_index_dict(), "tags": {"env": "prod"}}
    model = create_adapter.from_response(data)
    assert model.tags == {"env": "prod"}


# ---------------------------------------------------------------------------
# configure_adapter.to_request
# ---------------------------------------------------------------------------


def test_configure_adapter_serializes_empty() -> None:
    req = PreviewConfigureIndexRequest()
    raw = configure_adapter.to_request(req)
    assert raw == b"{}"


def test_configure_adapter_serializes_tags_only() -> None:
    req = PreviewConfigureIndexRequest(tags={"team": "infra"})
    raw = configure_adapter.to_request(req)
    data = orjson.loads(raw)
    assert data == {"tags": {"team": "infra"}}


# ---------------------------------------------------------------------------
# configure_adapter.from_response
# ---------------------------------------------------------------------------


def test_configure_adapter_parses_response() -> None:
    data = _minimal_index_dict("cfg-idx")
    model = configure_adapter.from_response(data)
    assert isinstance(model, PreviewIndexModel)
    assert model.name == "cfg-idx"
    assert model.host == "cfg-idx-host.pinecone.io"
    assert model.status.ready is True
    assert isinstance(model.deployment, PreviewManagedDeployment)


# ---------------------------------------------------------------------------
# describe_adapter.from_response
# ---------------------------------------------------------------------------


def test_describe_adapter_parses_response() -> None:
    data = _minimal_index_dict()
    model = describe_adapter.from_response(data)
    assert isinstance(model, PreviewIndexModel)
    assert model.name == "test-idx"
    assert model.status.ready is True
    assert model.deletion_protection == "disabled"


def test_describe_adapter_ignores_unknown_fields() -> None:
    data = _minimal_index_dict()
    data["private_host"] = "p"
    data["source_collection"] = "c"
    model = describe_adapter.from_response(data)
    assert model.name == "test-idx"


# ---------------------------------------------------------------------------
# list_adapter.from_response
# ---------------------------------------------------------------------------


def test_list_adapter_parses_empty() -> None:
    result = list_adapter.from_response({"indexes": []})
    assert result == []


def test_list_adapter_parses_multiple() -> None:
    data = {"indexes": [_minimal_index_dict("a"), _minimal_index_dict("b")]}
    result = list_adapter.from_response(data)
    assert len(result) == 2
    assert all(isinstance(m, PreviewIndexModel) for m in result)
    names = {m.name for m in result}
    assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# _filter_none edge cases
# ---------------------------------------------------------------------------


def test_create_adapter_preserves_zero_integer_value() -> None:
    req = PreviewCreateIndexRequest(
        schema={"fields": {"t": {"type": "string", "full_text_searchable": True, "max_term_len": 0}}}
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    assert data["schema"]["fields"]["t"]["max_term_len"] == 0


def test_filter_none_preserves_empty_dict_and_empty_list() -> None:
    result = _filter_none({"tags": {}, "ids": [], "name": None, "n": 0, "s": ""})
    assert result == {"tags": {}, "ids": [], "n": 0, "s": ""}


def test_filter_none_recurses_into_lists() -> None:
    result = _filter_none({"outer": [{"a": None, "b": 1}, {"a": None, "c": "x"}]})
    assert result == {"outer": [{"b": 1}, {"c": "x"}]}


def test_filter_none_preserves_false_in_nested_list() -> None:
    result = _filter_none({"items": [{"flag": False, "val": None}]})
    assert result == {"items": [{"flag": False}]}
