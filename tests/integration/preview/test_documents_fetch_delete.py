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
        {"_id": f"doc-{i}", "embedding": [float(i) / 10, 0.1, 0.2, 0.3]}
        for i in range(10)
    ]

    result = idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=5,
        max_workers=2,
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

    from pinecone.errors.exceptions import UnauthorizedError
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
        assert doc.category is not None, f"doc {doc_id} missing 'category' with include_fields=['*']"

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
        idx.documents.upsert(namespace="ns", documents=[{"_id": 42}])  # type: ignore[list-item]

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

    # delete by IDs must return None.
    result_ids = idx.documents.delete(namespace=ns, ids=["doc-0"])
    assert result_ids is None, (
        f"delete(ids=...) expected None, got {type(result_ids)}: {result_ids!r}"
    )

    # delete by filter must return None.
    result_filter = idx.documents.delete(
        namespace=ns, filter={"category": {"$eq": "vegetable"}}
    )
    assert result_filter is None, (
        f"delete(filter=...) expected None, got {type(result_filter)}: {result_filter!r}"
    )

    # delete_all=True must return None.
    result_all = idx.documents.delete(namespace=ns, delete_all=True)
    assert result_all is None, (
        f"delete(delete_all=True) expected None, got {type(result_all)}: {result_all!r}"
    )
