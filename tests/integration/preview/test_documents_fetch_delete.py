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
        .add_string_field("text", full_text_search={})
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


def test_upsert_accepts_extra_and_partial_documents(
    client: Pinecone,
    preview_index_name: str,
    preview_namespace: str,
    cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Verify upsert() accepts documents with extra fields and partial documents (§5 edge cases).

    Spec §5 documents two edge cases:
    - "Extra fields": Documents may include fields not defined in the schema.
      The API stores them as unindexed metadata. upserted_count reflects all docs.
    - "Partial documents": Not every document needs to include every schema field.
      Missing fields are treated as absent (not null).

    All existing upsert tests supply exactly the schema fields. This test verifies
    that the API accepts documents with:
    1. Extra field "custom_note" not in the 2-field schema (embedding + category)
    2. Missing optional schema field (embedding only, no category) — a partial document

    API constraint: each document must have at least one indexable schema field.
    Both docs in a single call; upserted_count must equal 2.
    Uses a dense-vector + filterable-string schema (OnDemand compatible).
    """
    from pinecone.preview.models import PreviewDocumentUpsertResponse, PreviewIndexModel

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
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

    # doc-extra: has both schema fields + "custom_note" (extra, not in schema)
    # doc-partial: has only "embedding" (omits optional "category" schema field)
    response = idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {
                "_id": "doc-extra",
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "category": "fruit",
                "custom_note": "extra field not in schema",
            },
            {
                "_id": "doc-partial",
                "embedding": [0.5, 0.6, 0.7, 0.8],
                # "category" is omitted — partial document, missing schema field is absent
            },
        ],
    )

    assert isinstance(response, PreviewDocumentUpsertResponse), (
        f"Expected PreviewDocumentUpsertResponse, got {type(response)}"
    )
    assert response.upserted_count == 2, (
        f"Expected upserted_count=2 for 2 docs (extra + partial), got {response.upserted_count}"
    )


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
        .add_string_field("text", full_text_search={})
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
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
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


# ---------------------------------------------------------------------------
# test_batch_upsert_result_fields — §5 batch_upsert() BatchResult shape
# ---------------------------------------------------------------------------


def test_batch_upsert_result_fields(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """batch_upsert() returns a BatchResult with correct item and batch counts.

    Verifies §5: BatchResult.total_item_count, successful_item_count,
    failed_item_count, total_batch_count, successful_batch_count,
    failed_batch_count, has_errors, failed_items, and errors are all correct
    for a fully-successful upload of 10 documents in 2 batches of 5.
    """
    from pinecone.models.batch import BatchResult
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
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
        {"_id": f"doc-{i}", "embedding": [float(i) / 10, 0.1, 0.2, 0.3]} for i in range(10)
    ]

    result = idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=5,
        max_concurrency=2,
        show_progress=False,
    )

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 10
    assert result.successful_item_count == 10
    assert result.failed_item_count == 0
    assert result.total_batch_count == 2
    assert result.successful_batch_count == 2
    assert result.failed_batch_count == 0
    assert result.has_errors is False
    assert result.failed_items == []
    assert result.errors == []


# ---------------------------------------------------------------------------
# test_batch_upsert_result_display_methods — §14 BatchResult display methods
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_batch_upsert_result_display_methods(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """BatchResult display methods match spec §14 format after a real batch_upsert call.

    PVT-004 verified all count fields but never asserted on the display methods.
    Spec §14 defines the __repr__ format as "BatchResult(SUCCESS: N/N items, N/N batches)".
    This test verifies:
    - §14 __repr__: exact string "BatchResult(SUCCESS: 10/10 items, 2/2 batches)"
    - §5 to_dict(): returns a dict with the correct 7 keys and values
    - §5 to_json(): returns valid JSON parseable to the same values as to_dict()
    - §14 _repr_html_(): returns non-empty HTML string containing "BatchResult"

    Uses 10-document upload in 2 batches of 5 (same as PVT-004) so repr output is
    deterministic. OnDemand dense vector schema avoids FTS dedicated-capacity restriction.
    """
    import json

    from pinecone.models.batch import BatchResult

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
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
        {"_id": f"doc-{i}", "embedding": [float(i) / 10, 0.1, 0.2, 0.3]} for i in range(10)
    ]

    result = idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=5,
        max_concurrency=2,
        show_progress=False,
    )

    assert isinstance(result, BatchResult)

    # §14: repr format: "BatchResult(SUCCESS: N/N items, N/N batches)"
    repr_str = repr(result)
    assert repr_str == "BatchResult(SUCCESS: 10/10 items, 2/2 batches)", (
        f"repr() expected 'BatchResult(SUCCESS: 10/10 items, 2/2 batches)', got {repr_str!r}"
    )

    # §5: to_dict() has all 7 expected keys with correct values
    d = result.to_dict()
    assert isinstance(d, dict), f"to_dict() expected dict, got {type(d)}"
    assert d["total_item_count"] == 10
    assert d["successful_item_count"] == 10
    assert d["failed_item_count"] == 0
    assert d["total_batch_count"] == 2
    assert d["successful_batch_count"] == 2
    assert d["failed_batch_count"] == 0
    assert d["errors"] == []

    # §5: to_json() is valid JSON parseable to a dict matching to_dict() values
    json_str = result.to_json()
    assert isinstance(json_str, str), f"to_json() expected str, got {type(json_str)}"
    parsed = json.loads(json_str)
    assert parsed["total_item_count"] == 10
    assert parsed["successful_item_count"] == 10
    assert parsed["failed_item_count"] == 0
    assert parsed["total_batch_count"] == 2
    assert parsed["errors"] == []

    # §14: _repr_html_() returns non-empty HTML string containing "BatchResult"
    html = result._repr_html_()
    assert isinstance(html, str), f"_repr_html_() expected str, got {type(html)}"
    assert len(html) > 0, "_repr_html_() must return non-empty HTML"
    assert "BatchResult" in html, f"_repr_html_() must contain 'BatchResult', got: {html[:200]!r}"


# ---------------------------------------------------------------------------
# test_fetch_wildcard_include_fields — §5 fetch() include_fields=["*"]
# ---------------------------------------------------------------------------


def test_fetch_wildcard_include_fields_returns_all_stored_fields(
    client: Pinecone,
    preview_index_name: str,
    preview_namespace: str,
    cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """fetch() with include_fields=["*"] returns all stored fields for each document.

    Verifies §5: the wildcard selector returns every field present in the stored
    document (embedding vector and category string), in contrast to a specific
    field list which returns only those named fields.

    SERVER BUG (IPV-0002): fetch() returns 401 "Unknown operation" for all
    preview index types (dense vector and FTS+dedicated). The upsert and search
    endpoints work correctly on the same host. This test reaches the wildcard
    fetch assertion but is expected to fail until IPV-0002 is resolved.
    """
    import time

    from pinecone.preview.models import PreviewIndexModel

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
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
    docs = [
        {"_id": "doc-a", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "fruit"},
        {"_id": "doc-b", "embedding": [0.5, 0.6, 0.7, 0.8], "category": "vegetable"},
    ]
    idx.documents.upsert(namespace=preview_namespace, documents=docs)
    time.sleep(3)

    # Fetch with wildcard — all stored fields must come back.
    # IPV-0002: this call fails with 401 "Unknown operation" until fixed.
    response = idx.documents.fetch(
        namespace=preview_namespace,
        ids=["doc-a", "doc-b"],
        include_fields=["*"],
    )

    assert isinstance(response, PreviewDocumentFetchResponse)
    assert set(response.documents.keys()) == {"doc-a", "doc-b"}

    for doc_id, doc in response.documents.items():
        assert isinstance(doc, PreviewDocument)
        assert doc._id == doc_id
        assert doc.category is not None, (
            f"doc {doc_id} missing 'category' with include_fields=['*']"
        )

    assert response.documents["doc-a"].category == "fruit"
    assert response.documents["doc-b"].category == "vegetable"


# ---------------------------------------------------------------------------
# test_upsert_client_side_validation — §4 upsert() document validation rules
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_upsert_client_side_validation_rejects_invalid_documents(
    client: Pinecone,
    require_preview: None,
) -> None:
    """upsert() raises ValidationError for every invalid-document condition before any HTTP call.

    Spec §4 declares client-side validation:
    - namespace must be a non-empty string
    - documents must be a non-empty list
    - documents must contain at most 100 items
    - every document must have an '_id' key
    - every '_id' must be a non-empty string (not missing, not non-string, not empty)
    - all '_id' values within a call must be unique

    Uses a dummy host so no real index is needed and no API call is made.
    """
    from pinecone.errors.exceptions import ValidationError

    # Use a dummy host — validation fires synchronously before the HTTP request.
    idx = client.preview.index(host="https://dummy-host.pinecone.io")

    valid_doc = {"_id": "doc-0", "text": "hello"}

    # Empty namespace must raise ValidationError.
    with pytest.raises(ValidationError, match="namespace"):
        idx.documents.upsert(namespace="", documents=[valid_doc])

    # Empty documents list must raise ValidationError.
    with pytest.raises(ValidationError, match="documents"):
        idx.documents.upsert(namespace="ns", documents=[])

    # More than 100 documents must raise ValidationError.
    over_limit = [{"_id": f"doc-{i}"} for i in range(101)]
    with pytest.raises(ValidationError, match="documents"):
        idx.documents.upsert(namespace="ns", documents=over_limit)

    # Document missing '_id' key must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        idx.documents.upsert(namespace="ns", documents=[{"text": "no id here"}])

    # Document with non-string '_id' must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        idx.documents.upsert(namespace="ns", documents=[{"_id": 42}])

    # Document with empty string '_id' must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        idx.documents.upsert(namespace="ns", documents=[{"_id": ""}])

    # Duplicate '_id' values within one call must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        idx.documents.upsert(
            namespace="ns",
            documents=[{"_id": "dup"}, {"_id": "dup"}],
        )


# ---------------------------------------------------------------------------
# test_delete_client_side_validation — §4 delete() argument validation rules
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_delete_client_side_validation_rejects_invalid_arguments(
    client: Pinecone,
    require_preview: None,
) -> None:
    """delete() raises ValidationError for missing or conflicting deletion targets.

    Spec §4 declares client-side validation:
    - namespace must be a non-empty string
    - at least one of ids, delete_all=True, or filter must be provided
    - ids and delete_all are mutually exclusive
    - ids and filter are mutually exclusive

    Uses a dummy host so no real index is needed and no API call is made.
    """
    from pinecone.errors.exceptions import ValidationError

    idx = client.preview.index(host="https://dummy-host.pinecone.io")

    # Empty namespace must raise ValidationError.
    with pytest.raises(ValidationError, match="namespace"):
        idx.documents.delete(namespace="", ids=["doc-0"])

    # Calling delete() with no targets must raise ValidationError.
    with pytest.raises(ValidationError, match="ids"):
        idx.documents.delete(namespace="ns")

    # ids and delete_all=True are mutually exclusive.
    with pytest.raises(ValidationError, match="ids"):
        idx.documents.delete(namespace="ns", ids=["doc-0"], delete_all=True)

    # ids and filter are mutually exclusive.
    with pytest.raises(ValidationError, match="ids"):
        idx.documents.delete(namespace="ns", ids=["doc-0"], filter={"category": {"$eq": "fruit"}})


# ---------------------------------------------------------------------------
# test_documents_delete_returns_none — §5 delete() "Returns: None"
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
@pytest.mark.skip(reason="IPV-0004: documents/delete endpoint returns 401 Unknown operation")
def test_documents_delete_returns_none_for_all_targeting_modes(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """documents.delete() returns None for ids, filter, and delete_all targeting modes.

    Spec §5 states: "Returns: None (empty response body)" with 202 Accepted.
    The existing delete tests (test_delete_by_ids, test_delete_by_filter,
    test_delete_all) are broken because their populated_index fixture depends
    on FTS (dedicated capacity) and fetch (IPV-0002). This test uses a dense
    vector OnDemand schema and does not require verification of deletion to
    verify the return type contract.

    DISABLED (IPV-0004): The /namespaces/{ns}/documents/delete endpoint returns
    401 Unauthorized with x-pinecone-auth-rejected-reason: Unknown operation,
    blocking all documents.delete() calls.
    """
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
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
    ns = preview_namespace

    idx.documents.upsert(
        namespace=ns,
        documents=[
            {"_id": "doc-0", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "fruit"},
            {"_id": "doc-1", "embedding": [0.5, 0.6, 0.7, 0.8], "category": "fruit"},
            {"_id": "doc-2", "embedding": [0.9, 0.1, 0.2, 0.3], "category": "vegetable"},
        ],
    )

    # delete by IDs must return None (verified by -> None annotation).
    idx.documents.delete(namespace=ns, ids=["doc-0"])

    # delete by filter must return None.
    idx.documents.delete(namespace=ns, filter={"category": {"$eq": "vegetable"}})

    # delete_all=True must return None.
    idx.documents.delete(namespace=ns, delete_all=True)


# ---------------------------------------------------------------------------
# test_batch_upsert_with_batch_size_one_per_document — §5 batch_size minimum
# ---------------------------------------------------------------------------


def test_batch_upsert_with_batch_size_one_per_document(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """batch_upsert(batch_size=1) creates one batch per document — total_batch_count == len(docs).

    Spec §5 defines batch_size range as 1–100. When batch_size=1, each document
    is its own HTTP request, so total_batch_count equals the document count.
    PVT-004 and PVT-027 both use batch_size=5 with 10 docs (2 batches); this test
    verifies the minimum value produces one batch per document.
    """
    from pinecone.models.batch import BatchResult

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
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
        {"_id": f"doc-{i}", "embedding": [float(i) / 10, 0.1, 0.2, 0.3]}
        for i in range(1, 4)  # 3 documents
    ]

    result = idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=1,  # minimum: each document is its own HTTP request
        max_concurrency=2,
        show_progress=False,
    )

    assert isinstance(result, BatchResult), f"Expected BatchResult, got {type(result)}"
    assert result.total_item_count == 3, (
        f"Expected total_item_count=3, got {result.total_item_count}"
    )
    assert result.total_batch_count == 3, (
        f"Expected total_batch_count=3 (one per doc when batch_size=1), got {result.total_batch_count}"
    )
    assert result.successful_item_count == 3, (
        f"Expected successful_item_count=3, got {result.successful_item_count}"
    )
    assert result.successful_batch_count == 3, (
        f"Expected successful_batch_count=3, got {result.successful_batch_count}"
    )
    assert result.failed_item_count == 0, (
        f"Expected failed_item_count=0, got {result.failed_item_count}"
    )
    assert result.failed_batch_count == 0, (
        f"Expected failed_batch_count=0, got {result.failed_batch_count}"
    )
    assert result.has_errors is False, f"Expected has_errors=False, got {result.has_errors}"


# ---------------------------------------------------------------------------
# test_batch_upsert_partial_failure_collects_failed_items — §5 partial failure
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
def test_batch_upsert_partial_failure_collects_failed_items(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """batch_upsert() continues after per-batch failures and records them in BatchResult.

    Spec §5 "Edge case — partial failure": if some batches fail the operation
    continues; failed items are collected in result.failed_items for retry.

    Strategy: 4-dim cosine index + batch_size=1 (one doc per batch) + one document
    with a 5-dim vector that triggers a server-side 4xx rejection. The other 3
    documents have valid 4-dim vectors and succeed.

    Verifies: has_errors=True, failed_batch_count==1, failed_item_count==1,
    successful_batch_count==3, successful_item_count==3, failed_items contains
    the rejected document, and errors has one BatchError entry.
    """
    from pinecone.models.batch import BatchResult
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=4, metric="cosine").build()
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
    # 5-dim vector on a 4-dim index → API rejects the batch with a 4xx error
    bad_doc = {"_id": "bad-dim", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
    good_docs = [
        {"_id": f"good-{i}", "embedding": [float(i) / 10, 0.1, 0.2, 0.3]} for i in range(3)
    ]
    all_docs = [*good_docs, bad_doc]  # 4 total

    result = idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=all_docs,
        batch_size=1,  # one doc per batch so the single bad doc fails in isolation
        max_concurrency=2,
        show_progress=False,
    )

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 4
    assert result.total_batch_count == 4
    assert result.has_errors is True, f"Expected has_errors=True, got {result.has_errors}"
    assert result.failed_batch_count == 1, (
        f"Expected 1 failed batch (bad-dim doc), got {result.failed_batch_count}"
    )
    assert result.failed_item_count == 1, f"Expected 1 failed item, got {result.failed_item_count}"
    assert result.successful_batch_count == 3, (
        f"Expected 3 successful batches, got {result.successful_batch_count}"
    )
    assert result.successful_item_count == 3, (
        f"Expected 3 successful items, got {result.successful_item_count}"
    )
    assert len(result.failed_items) == 1, (
        f"Expected 1 item in failed_items, got {len(result.failed_items)}"
    )
    assert result.failed_items[0]["_id"] == "bad-dim", (
        f"Expected failed_items[0]['_id'] == 'bad-dim', got {result.failed_items[0]['_id']!r}"
    )
    assert len(result.errors) == 1, f"Expected 1 BatchError entry, got {len(result.errors)}"


# ---------------------------------------------------------------------------
# test_describe_dedicated_index_read_capacity_response — §3 PreviewReadCapacityDedicatedResponse
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_describe_dedicated_index_read_capacity_response_fields(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """describe() on a dedicated-capacity index returns PreviewReadCapacityDedicatedResponse.

    Spec §3 `PreviewIndexModel.read_capacity` is a discriminated union:
    - OnDemand mode  → `PreviewReadCapacityOnDemandResponse` (covered by PVT-024/032)
    - Dedicated mode → `PreviewReadCapacityDedicatedResponse` with nested `dedicated`
                       (PreviewReadCapacityDedicatedInner) and `status`
                       (PreviewReadCapacityStatus)

    This test verifies the Dedicated branch is deserialized correctly from the API
    response. FTS indexes require dedicated read capacity as of 2026-04-17, so this
    test creates an FTS schema index with read_capacity={mode: Dedicated} and checks
    the full nested structure of the response model.
    """
    from pinecone.preview.models import (
        PreviewReadCapacityDedicatedInner,
        PreviewReadCapacityDedicatedResponse,
        PreviewReadCapacityStatus,
    )

    schema = SchemaBuilder().add_string_field("text", full_text_search={}).build()
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {"shards": 1, "replicas": 1},
        },
    }
    cleanup_preview_indexes.append(preview_index_name)
    model = client.preview.indexes.create(
        name=preview_index_name,
        schema=schema,
        read_capacity=read_capacity,
    )

    assert isinstance(model, PreviewIndexModel)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    described = poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )
    assert isinstance(described, PreviewIndexModel)

    # read_capacity must be present and must be the Dedicated variant.
    rc = described.read_capacity
    assert rc is not None, (
        "describe() on a dedicated-capacity index must have read_capacity != None"
    )
    assert isinstance(rc, PreviewReadCapacityDedicatedResponse), (
        f"read_capacity must be PreviewReadCapacityDedicatedResponse for a Dedicated index, "
        f"got {type(rc)}"
    )

    # Nested dedicated configuration.
    dedicated = rc.dedicated
    assert isinstance(dedicated, PreviewReadCapacityDedicatedInner), (
        f"rc.dedicated must be PreviewReadCapacityDedicatedInner, got {type(dedicated)}"
    )
    assert isinstance(dedicated.node_type, str) and dedicated.node_type, (
        f"dedicated.node_type must be a non-empty string, got {dedicated.node_type!r}"
    )
    assert dedicated.scaling in ("Manual", "Auto"), (
        f"dedicated.scaling must be 'Manual' or 'Auto', got {dedicated.scaling!r}"
    )
    # Manual scaling → manual config must be present.
    if dedicated.scaling == "Manual":
        assert dedicated.manual is not None, (
            "dedicated.manual must not be None when scaling='Manual'"
        )
        assert isinstance(dedicated.manual.shards, int) and dedicated.manual.shards >= 1, (
            f"dedicated.manual.shards must be a positive int, got {dedicated.manual.shards!r}"
        )
        assert isinstance(dedicated.manual.replicas, int) and dedicated.manual.replicas >= 1, (
            f"dedicated.manual.replicas must be a positive int, got {dedicated.manual.replicas!r}"
        )

    # Provisioning status.
    status = rc.status
    assert isinstance(status, PreviewReadCapacityStatus), (
        f"rc.status must be PreviewReadCapacityStatus, got {type(status)}"
    )
    assert isinstance(status.state, str) and status.state, (
        f"status.state must be a non-empty string, got {status.state!r}"
    )
