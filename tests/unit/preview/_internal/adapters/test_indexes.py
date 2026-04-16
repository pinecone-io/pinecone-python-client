"""Unit tests for preview index adapters."""

from __future__ import annotations

import orjson

from pinecone.preview._internal.adapters.indexes import (
    configure_adapter,
    create_adapter,
    describe_adapter,
    list_adapter,
)
from pinecone.preview.models.deployment import PreviewManagedDeployment
from pinecone.preview.models.indexes import PreviewIndexModel
from pinecone.preview.models.read_capacity import (
    PreviewReadCapacityOnDemandResponse,
    PreviewReadCapacityStatus,
)
from pinecone.preview.models.requests import PreviewConfigureIndexRequest, PreviewCreateIndexRequest
from pinecone.preview.models.schema import (
    PreviewDenseVectorField,
    PreviewSchema,
    PreviewStringField,
)


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
        schema=PreviewSchema(fields={"e": PreviewDenseVectorField(dimension=4)})
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    assert set(data.keys()) == {"schema"}
    assert data["schema"]["fields"]["e"]["type"] == "dense_vector"


def test_create_adapter_preserves_explicit_false_overrides() -> None:
    req = PreviewCreateIndexRequest(
        schema=PreviewSchema(
            fields={
                "title": PreviewStringField(filterable=False, full_text_searchable=True)
            }
        )
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    field = data["schema"]["fields"]["title"]
    assert field.get("filterable") is False
    assert field.get("full_text_searchable") is True


def test_create_adapter_preserves_user_explicit_lowercase_false() -> None:
    req = PreviewCreateIndexRequest(
        schema=PreviewSchema(
            fields={
                "title": PreviewStringField(full_text_searchable=True, lowercase=False)
            }
        )
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    field = data["schema"]["fields"]["title"]
    assert field.get("lowercase") is False
    assert field.get("full_text_searchable") is True


def test_create_adapter_preserves_stemming_false_and_stop_words_false() -> None:
    req = PreviewCreateIndexRequest(
        schema=PreviewSchema(
            fields={
                "title": PreviewStringField(full_text_searchable=True, stemming=False, stop_words=False)
            }
        )
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    field = data["schema"]["fields"]["title"]
    assert field.get("stemming") is False
    assert field.get("stop_words") is False
    assert field.get("full_text_searchable") is True


def test_create_adapter_serializes_full_request() -> None:
    req = PreviewCreateIndexRequest(
        schema=PreviewSchema(fields={"e": PreviewDenseVectorField(dimension=4)}),
        name="my-idx",
        deployment=PreviewManagedDeployment(
            environment="us-east-1-aws", cloud="aws", region="us-east-1"
        ),
        read_capacity=PreviewReadCapacityOnDemandResponse(
            status=PreviewReadCapacityStatus(state="Ready")
        ),
        tags={"env": "prod"},
    )
    raw = create_adapter.to_request(req)
    data = orjson.loads(raw)
    assert data["name"] == "my-idx"
    assert data["tags"] == {"env": "prod"}
    assert data["deployment"]["deployment_type"] == "managed"


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
