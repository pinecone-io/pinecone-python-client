"""Integration tests for integrated embedding on document indexes.

Each module-scoped fixture creates a throwaway document index whose ``string``
field declares ``embed`` / ``sparse_embed``, upserts plain text, and deletes the
index afterwards. Searches poll because upserts become visible asynchronously.

The project must allow ``embed`` / ``sparse_embed`` on create. Where it does not,
create fails with the server's 400 and these tests fail rather than skip.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pinecone import Pinecone
from pinecone.errors import ApiError
from pinecone.index import Index
from pinecone.models.documents.responses import SearchDocumentsResponse
from pinecone.models.documents.score_by import (
    DocumentScoringMethod,
    EmbedQuery,
    SparseEmbedQuery,
    TextQuery,
)
from pinecone.schema_builder import SchemaBuilder
from tests.integration.conftest import ensure_index_deleted, poll_until, unique_name

pytestmark = [pytest.mark.integration]

NAMESPACE = "embed-it"
DENSE_MODEL = "multilingual-e5-large"
CORPUS = [
    {"_id": "d1", "body": "the quick brown fox jumps over the lazy dog", "genre": "animal"},
    {"_id": "d2", "body": "pinecone serverless vector database", "genre": "tech"},
    {"_id": "d3", "body": "a slow grey wolf sleeps under the old oak tree", "genre": "animal"},
    {"_id": "d4", "body": "managed vector search service for developers", "genre": "tech"},
]


@pytest.fixture(scope="module")
def stored_index(client: Pinecone) -> Iterator[Index]:
    name = unique_name("embed-docs")
    schema = (
        SchemaBuilder()
        .add_string_field(
            "body",
            full_text_search=True,
            embed={"model": DENSE_MODEL},
            sparse_embed=True,
        )
        .build()
    )
    try:
        client.indexes.create(name=name, schema=schema, timeout=300)
        index = client.index(name=name)
        index.documents.upsert(namespace=NAMESPACE, documents=CORPUS)
        yield index
        index.close()
    finally:
        ensure_index_deleted(client, name)


@pytest.fixture(scope="module")
def unstored_index(client: Pinecone) -> Iterator[Index]:
    name = unique_name("embed-unstored")
    schema = (
        SchemaBuilder()
        .add_string_field("body", embed={"model": DENSE_MODEL}, store_text=False)
        .build()
    )
    try:
        client.indexes.create(name=name, schema=schema, timeout=300)
        index = client.index(name=name)
        index.documents.upsert(namespace=NAMESPACE, documents=CORPUS[:1])
        yield index
        index.close()
    finally:
        ensure_index_deleted(client, name)


def _top_hit(index: Index, clause: DocumentScoringMethod, expected: str) -> SearchDocumentsResponse:
    result = poll_until(
        lambda: index.documents.search(
            namespace=NAMESPACE, top_k=4, score_by=[clause], include_fields=["genre"]
        ),
        lambda r: bool(r.matches) and r.matches[0].id == expected,
        timeout=120,
        description=f"{type(clause).__name__} ranking {expected!r} first",
    )
    assert isinstance(result, SearchDocumentsResponse)
    return result


def test_embed_search_ranks_the_closest_document_first(stored_index: Index) -> None:
    result = _top_hit(
        stored_index, EmbedQuery(field="body", query="a fox leaping over a dog"), "d1"
    )

    assert result.matches[0].genre == "animal"
    assert result.usage is not None and result.usage.read_units >= 0


def test_sparse_embed_search_ranks_the_closest_document_first(stored_index: Index) -> None:
    _top_hit(stored_index, SparseEmbedQuery(field="body", query="grey wolf oak tree"), "d3")


def test_text_search_on_an_embedded_field(stored_index: Index) -> None:
    _top_hit(stored_index, TextQuery(fields=["body"], query="serverless"), "d2")


def test_include_fields_naming_an_unstored_field_is_rejected(unstored_index: Index) -> None:
    with pytest.raises(ApiError) as excinfo:
        unstored_index.documents.fetch(namespace=NAMESPACE, ids=["d1"], include_fields=["body"])

    assert excinfo.value.status_code == 400
    assert "store_text" in str(excinfo.value)
