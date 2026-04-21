"""Unit tests for IndexModel.to_dict() recursive conversion."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models.indexes.index import (
    IndexModel,
    IndexSpec,
    IndexStatus,
    ModelIndexEmbed,
    ServerlessSpecInfo,
)


def _make_serverless_index(**kwargs: object) -> IndexModel:
    return IndexModel(
        name="idx",
        metric="cosine",
        host="localhost",
        status=IndexStatus(ready=True, state="Ready"),
        spec=IndexSpec(serverless=ServerlessSpecInfo(cloud="aws", region="us-east-1")),
        **kwargs,  # type: ignore[arg-type]
    )


def test_to_dict_required_fields_only() -> None:
    model = _make_serverless_index()
    result = model.to_dict()

    assert isinstance(result["status"], dict)
    assert "ready" in result["status"]
    assert "state" in result["status"]

    assert isinstance(result["spec"], dict)
    assert "serverless" in result["spec"]


def test_to_dict_nested_struct_recursive() -> None:
    model = _make_serverless_index()
    result = model.to_dict()

    assert not isinstance(result["status"], Struct)
    assert not isinstance(result["spec"], Struct)
    assert not isinstance(result["spec"]["serverless"], Struct)


def test_to_dict_embed_present() -> None:
    model = _make_serverless_index(embed=ModelIndexEmbed(model="ml-e5", metric="cosine"))
    result = model.to_dict()

    assert isinstance(result["embed"], dict)
    assert result["embed"]["model"] == "ml-e5"


def test_to_dict_embed_none() -> None:
    model = _make_serverless_index()
    result = model.to_dict()

    assert result["embed"] is None


def test_to_dict_tags_preserved() -> None:
    model = _make_serverless_index(tags={"env": "prod"})
    result = model.to_dict()

    assert result["tags"] == {"env": "prod"}


def test_to_dict_is_pure_read() -> None:
    model = _make_serverless_index()
    d = model.to_dict()
    d["name"] = "mutated"
    second = model.to_dict()

    assert second["name"] == "idx"
