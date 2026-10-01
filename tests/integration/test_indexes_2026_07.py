"""Integration tests for the graduated 2026-07 sync index lifecycle (#131).

Runs against whatever PINECONE_API_KEY / PINECONE_CONTROLLER_HOST point to —
a real project or a local minicone (`PINECONE_API_KEY=mockkey
PINECONE_CONTROLLER_HOST=http://127.0.0.1:5080`). Until #112 flips
CONTROL_PLANE_API_VERSION to 2026-07, version-dispatching servers will
reject these requests; that skew is the planned intermediate state.

Migrated from tests/integration/preview/test_indexes_lifecycle.py onto the
graduated names; the remaining spec-shape migration of
tests/integration/test_indexes.py belongs to #174.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Generator
from typing import Any

import msgspec
import orjson
import pytest

from pinecone import Pinecone
from pinecone._internal.adapters.indexes_adapter import IndexesAdapter
from pinecone.errors import ApiError, NotFoundError, PineconeError
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.requests import CreateIndexRequest
from pinecone.models.indexes.schema import (
    FieldEmbedConfig,
    FieldSparseEmbedConfig,
    IndexSchema,
    StringField,
)
from pinecone.schema_builder import SchemaBuilder

pytestmark = [pytest.mark.integration]

_DENSE_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 4, "metric": "cosine"}}
}


@pytest.fixture
def index_name(client: Pinecone) -> Generator[str, None, None]:
    name = f"g131-{uuid.uuid4().hex[:10]}"
    yield name
    with contextlib.suppress(NotFoundError, PineconeError):
        client.indexes.configure(name, deletion_protection="disabled")
    with contextlib.suppress(NotFoundError, PineconeError):
        client.indexes.delete(name, timeout=120)


def test_lifecycle_create_describe_list_configure_delete(client: Pinecone, index_name: str) -> None:
    created = client.indexes.create(
        name=index_name,
        schema=_DENSE_SCHEMA,
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
        tags={"suite": "g131"},
        timeout=300,
    )
    assert isinstance(created, IndexModel)
    assert created.name == index_name
    assert created.status.ready is True
    assert "embedding" in created.schema.fields

    described = client.indexes.describe(index_name)
    assert described.host
    assert client.indexes.exists(index_name) is True

    names = [idx.name for idx in client.indexes.list()]
    assert index_name in names

    updated = client.indexes.configure(index_name, tags={"suite": "g131", "phase": "two"})
    assert isinstance(updated, IndexModel)

    protected = client.indexes.configure(index_name, deletion_protection="enabled")
    assert protected.deletion_protection == "enabled"
    unprotected = client.indexes.configure(index_name, deletion_protection="disabled")
    assert unprotected.deletion_protection == "disabled"

    client.indexes.delete(index_name, timeout=120)
    assert client.indexes.exists(index_name) is False


def test_create_fts_index(client: Pinecone, index_name: str) -> None:
    created = client.indexes.create(
        name=index_name,
        schema={
            "fields": {
                "body": {"type": "string", "full_text_search": {"language": "en"}},
            }
        },
        timeout=300,
    )
    assert created.status.ready is True
    assert "body" in created.schema.fields


def test_create_fts_index_from_builder(client: Pinecone, index_name: str) -> None:
    schema = SchemaBuilder().add_string_field("body", language="en").build()
    created = client.indexes.create(name=index_name, schema=schema, timeout=300)
    assert created.status.ready is True
    assert "body" in created.schema.fields


def test_create_without_name_assigns_one(client: Pinecone) -> None:
    created = client.indexes.create(schema=_DENSE_SCHEMA, timeout=300)
    try:
        assert created.name
        assert client.indexes.exists(created.name) is True
    finally:
        with contextlib.suppress(NotFoundError, PineconeError):
            client.indexes.delete(created.name, timeout=120)


def test_bare_string_field_rejected_by_server(client: Pinecone, index_name: str) -> None:
    """A string field is accepted only with ``full_text_search``; the two
    unaccepted shapes are refused at different layers.

    ``{"type": "string"}`` matches neither variant of the server's untagged
    ``CreateStringSchemaField`` union — ``full_text_search`` is missing for the
    text-search variant, ``filterable`` is a required bool on the metadata
    variant — so it fails deserialization with 422 and never reaches schema
    validation, which is where the documented "fields used for search" 400
    lives. The two rejections are each other's positive control: a blanket
    refusal of index creation would answer both with one identical status.
    """
    with pytest.raises(ApiError) as bare:
        client.indexes.create(
            name=index_name, schema={"fields": {"title": {"type": "string"}}}, timeout=-1
        )
    assert bare.value.status_code == 422

    with pytest.raises(ApiError) as filter_only:
        client.indexes.create(
            name=index_name,
            schema={
                "fields": {
                    "embedding": {"type": "dense_vector", "dimension": 4, "metric": "cosine"},
                    "title": {"type": "string", "filterable": True},
                }
            },
            timeout=-1,
        )
    assert filter_only.value.status_code == 400
    assert "only accepts fields used for search" in str(filter_only.value)


def test_create_and_describe_embed_and_sparse_embed_fields(
    client: Pinecone, index_name: str
) -> None:
    """One index covers ``embed``, ``sparse_embed`` and ``store_text: false``.

    Describe resolves every omitted embed key from the model, reports the
    full-text-search config a stored field gets by default, and leaves it off
    the unstored field. The described schema converts back into the typed model
    and into a create body without ``filterable``.
    """
    schema = (
        SchemaBuilder()
        .add_string_field("title", embed=True)
        .add_string_field("body", sparse_embed=True, store_text=False)
        .build()
    )
    created = client.indexes.create(name=index_name, schema=schema, timeout=300)
    assert created.status.ready is True

    described = client.indexes.describe(index_name)
    title = described.schema.fields["title"]
    body = described.schema.fields["body"]
    assert isinstance(title, StringField)
    assert isinstance(body, StringField)

    assert isinstance(title.embed, FieldEmbedConfig)
    assert title.embed.model
    assert isinstance(title.embed.dimension, int) and title.embed.dimension > 0
    assert title.embed.metric in {"cosine", "dotproduct", "euclidean"}
    assert title.embed.write_parameters
    assert title.embed.read_parameters
    assert title.sparse_embed is None
    assert title.store_text is True
    assert title.full_text_search is not None

    assert isinstance(body.sparse_embed, FieldSparseEmbedConfig)
    assert body.sparse_embed.model
    assert body.sparse_embed.write_parameters
    assert body.sparse_embed.read_parameters
    assert body.embed is None
    assert body.store_text is False
    assert body.full_text_search is None

    assert msgspec.convert(described.schema.to_dict(), type=IndexSchema) == described.schema
    resent = orjson.loads(
        IndexesAdapter.to_create_request(
            CreateIndexRequest(name=index_name, schema=described.schema)
        )
    )["schema"]["fields"]
    assert resent["title"]["embed"]["model"] == title.embed.model
    assert resent["body"]["store_text"] is False
    assert "filterable" not in resent["title"]
    assert "filterable" not in resent["body"]
