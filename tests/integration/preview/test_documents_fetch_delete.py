"""Integration tests for preview document fetch and delete flows (2026-01.alpha).

Covers §13 end-to-end examples from spec/preview.md:
- Fetch documents by ID (dict keyed by _id, silent omission of missing IDs)
- Fetch documents by metadata filter
- Delete by IDs, by filter, and delete_all
- Preview control-plane to stable data-plane interop (§13 "interop")

These tests make real API calls and skip gracefully when the preview
endpoint is unavailable. They do NOT gate CI (preview_integration marker).

Eventual consistency: fetch results are eventually consistent after upsert.
Each test polls via poll_until() with appropriate timeouts.
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.preview import SchemaBuilder
from pinecone.preview.models import (
    PreviewDocument,
    PreviewDocumentFetchResponse,
    PreviewIndexModel,
)
from tests.integration.conftest import poll_until

pytestmark = [pytest.mark.integration, pytest.mark.preview_integration]

_DOCS = [
    {"_id": "doc-0", "text": "apple", "category": "fruit"},
    {"_id": "doc-1", "text": "banana", "category": "fruit"},
    {"_id": "doc-2", "text": "carrot", "category": "vegetable"},
    {"_id": "doc-3", "text": "orange", "category": "fruit"},
    {"_id": "doc-4", "text": "spinach", "category": "vegetable"},
]
_ALL_IDS = {d["_id"] for d in _DOCS}


# ---------------------------------------------------------------------------
# Local fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_index(
    client: Pinecone,
    preview_index_name: str,
    preview_namespace: str,
    cleanup_preview_indexes: list[str],
    require_preview: None,
) -> tuple[str, str]:
    """Create a small FTS+filterable index and upsert 5 deterministic docs.

    Returns (index_name, namespace). Cleanup is handled by cleanup_preview_indexes.
    """
    schema = (
        SchemaBuilder()
        .add_string_field("text", full_text_searchable=True)
        .add_string_field("category", filterable=True)
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    idx.documents.upsert(namespace=preview_namespace, documents=_DOCS)

    # Poll until all 5 docs are fetchable (eventual consistency).
    def _query() -> PreviewDocumentFetchResponse:
        return idx.documents.fetch(
            namespace=preview_namespace,
            ids=list(_ALL_IDS),
            include_fields=["text"],
        )

    def _all_present(r: PreviewDocumentFetchResponse) -> bool:
        return _ALL_IDS.issubset(r.documents.keys())

    poll_until(
        _query,
        _all_present,
        timeout=120,
        interval=3,
        description="all 5 docs fetchable",
    )

    return (preview_index_name, preview_namespace)


# ---------------------------------------------------------------------------
# Fetch tests
# ---------------------------------------------------------------------------


def test_fetch_by_ids_returns_dict_keyed_by_id(
    client: Pinecone,
    populated_index: tuple[str, str],
) -> None:
    """Fetch by IDs returns PreviewDocumentFetchResponse.documents keyed by _id."""
    idx = client.preview.index(name=populated_index[0])
    response = idx.documents.fetch(
        namespace=populated_index[1],
        ids=["doc-0", "doc-1", "doc-2"],
        include_fields=["text"],
    )

    assert isinstance(response, PreviewDocumentFetchResponse)
    assert set(response.documents.keys()) == {"doc-0", "doc-1", "doc-2"}

    expected_texts = {"doc-0": "apple", "doc-1": "banana", "doc-2": "carrot"}
    for doc_id, expected_text in expected_texts.items():
        doc = response.documents[doc_id]
        assert isinstance(doc, PreviewDocument)
        assert doc._id == doc_id
        assert doc.text == expected_text


def test_fetch_missing_id_silently_omitted(
    client: Pinecone,
    populated_index: tuple[str, str],
) -> None:
    """Missing IDs in a fetch request are silently omitted, no exception raised."""
    idx = client.preview.index(name=populated_index[0])
    response = idx.documents.fetch(
        namespace=populated_index[1],
        ids=["doc-0", "doc-99999", "doc-1"],
        include_fields=["text"],
    )

    assert "doc-99999" not in response.documents
    assert {"doc-0", "doc-1"} <= set(response.documents.keys())


def test_fetch_by_filter_returns_matching_docs(
    client: Pinecone,
    populated_index: tuple[str, str],
) -> None:
    """Fetch by metadata filter returns only documents matching the filter."""
    idx = client.preview.index(name=populated_index[0])
    response = idx.documents.fetch(
        namespace=populated_index[1],
        filter={"category": {"$eq": "fruit"}},
        include_fields=["text", "category"],
    )

    assert isinstance(response, PreviewDocumentFetchResponse)
    for doc in response.documents.values():
        assert doc.category == "fruit", f"Expected fruit, got {doc.category!r}"
    assert {"doc-0", "doc-1", "doc-3"} <= set(response.documents.keys())


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------


def test_delete_by_ids(
    client: Pinecone,
    populated_index: tuple[str, str],
) -> None:
    """Delete by IDs removes only the specified documents."""
    idx = client.preview.index(name=populated_index[0])
    namespace = populated_index[1]

    idx.documents.delete(namespace=namespace, ids=["doc-0", "doc-2"])

    def _fetch_deleted() -> PreviewDocumentFetchResponse:
        return idx.documents.fetch(
            namespace=namespace,
            ids=["doc-0", "doc-2"],
            include_fields=["text"],
        )

    def _both_absent(r: PreviewDocumentFetchResponse) -> bool:
        return "doc-0" not in r.documents and "doc-2" not in r.documents

    poll_until(
        _fetch_deleted,
        _both_absent,
        timeout=90,
        interval=3,
        description="doc-0 and doc-2 deleted",
    )

    # doc-1 is untouched
    surviving = idx.documents.fetch(
        namespace=namespace,
        ids=["doc-1"],
        include_fields=["text"],
    )
    assert surviving.documents["doc-1"]._id == "doc-1"


def test_delete_by_filter(
    client: Pinecone,
    populated_index: tuple[str, str],
) -> None:
    """Delete by filter removes all documents matching the filter expression."""
    idx = client.preview.index(name=populated_index[0])
    namespace = populated_index[1]

    idx.documents.delete(namespace=namespace, filter={"category": {"$eq": "vegetable"}})

    def _fetch_vegetables() -> PreviewDocumentFetchResponse:
        return idx.documents.fetch(
            namespace=namespace,
            filter={"category": {"$eq": "vegetable"}},
            include_fields=["category"],
        )

    def _no_vegetables(r: PreviewDocumentFetchResponse) -> bool:
        return len(r.documents) == 0

    poll_until(
        _fetch_vegetables,
        _no_vegetables,
        timeout=90,
        interval=3,
        description="all vegetable docs deleted",
    )

    # Fruit docs remain
    fruits = idx.documents.fetch(
        namespace=namespace,
        filter={"category": {"$eq": "fruit"}},
        include_fields=["category"],
    )
    assert len(fruits.documents) > 0


def test_delete_all(
    client: Pinecone,
    populated_index: tuple[str, str],
) -> None:
    """delete_all=True removes all documents in the namespace."""
    idx = client.preview.index(name=populated_index[0])
    namespace = populated_index[1]

    idx.documents.delete(namespace=namespace, delete_all=True)

    def _fetch_all() -> PreviewDocumentFetchResponse:
        return idx.documents.fetch(
            namespace=namespace,
            ids=["doc-0", "doc-1", "doc-2", "doc-3", "doc-4"],
            include_fields=["text"],
        )

    def _all_absent(r: PreviewDocumentFetchResponse) -> bool:
        return len(r.documents) == 0

    poll_until(
        _fetch_all,
        _all_absent,
        timeout=90,
        interval=3,
        description="all docs deleted",
    )


# ---------------------------------------------------------------------------
# Interop test — §13 "Preview control plane to stable data plane"
# ---------------------------------------------------------------------------


def test_preview_control_plane_to_stable_data_plane_interop(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Preview-created index host is usable by both preview and stable SDK channels.

    Verifies that model.host from a preview create call is a plain URL accepted
    by both pc.preview.index(host=...) and pc.index(host=...) without modification.
    The stable query API (ScoredVector.id) and the preview fetch API (_id) address
    the same document.
    """
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("text", full_text_searchable=True)
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    model = client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    assert isinstance(model, PreviewIndexModel)
    assert model.host

    # Preview data-plane: upsert via preview channel
    preview_idx = client.preview.index(host=model.host)
    preview_idx.documents.upsert(
        namespace="ns",
        documents=[{"_id": "doc-1", "embedding": [0.1, 0.2, 0.3, 0.4], "text": "hello world"}],
    )

    # Poll preview fetch until doc-1 is visible
    def _fetch_doc1() -> PreviewDocumentFetchResponse:
        return preview_idx.documents.fetch(
            namespace="ns",
            ids=["doc-1"],
            include_fields=["text"],
        )

    def _doc1_present(r: PreviewDocumentFetchResponse) -> bool:
        return "doc-1" in r.documents

    poll_until(
        _fetch_doc1,
        _doc1_present,
        timeout=120,
        interval=3,
        description="doc-1 visible via preview fetch",
    )

    # Stable data-plane: query via stable channel against the same host
    from pinecone.models.vectors.responses import QueryResponse

    stable_idx = client.index(host=model.host)

    def _stable_query() -> QueryResponse:
        return stable_idx.query(
            vector=[0.1, 0.2, 0.3, 0.4],
            top_k=1,
            namespace="ns",
        )

    def _has_match(r: QueryResponse) -> bool:
        return len(r.matches) > 0

    results = poll_until(
        _stable_query,
        _has_match,
        timeout=90,
        interval=3,
        description="stable query returns doc-1",
    )

    assert isinstance(results, QueryResponse)
    assert results.matches[0].id == "doc-1"


# ---------------------------------------------------------------------------
# test_upsert_returns_upserted_count — §5 upsert() response shape
# ---------------------------------------------------------------------------


def test_upsert_returns_upserted_count(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """upsert() returns PreviewDocumentUpsertResponse with upserted_count == len(documents).

    Verifies §5: the response object carries upserted_count equal to the number
    of documents sent in the request. This is distinct from existing tests that
    call upsert() as setup but never assert the return value.
    """
    from pinecone.preview.models import PreviewDocumentUpsertResponse, PreviewIndexModel

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = client.preview.index(name=preview_index_name)
    documents = [
        {"_id": "doc-0", "embedding": [0.1, 0.2, 0.3, 0.4]},
        {"_id": "doc-1", "embedding": [0.5, 0.6, 0.7, 0.8]},
        {"_id": "doc-2", "embedding": [0.9, 0.1, 0.2, 0.3]},
    ]
    response = idx.documents.upsert(namespace=preview_namespace, documents=documents)

    assert isinstance(response, PreviewDocumentUpsertResponse)
    assert response.upserted_count == len(documents)
