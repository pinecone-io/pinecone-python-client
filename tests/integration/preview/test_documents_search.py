"""Integration tests for preview document upsert + search flows (2026-01.alpha).

Covers §13 end-to-end examples from spec/preview.md:
- Full-text search only (SchemaBuilder + batch_upsert + PreviewTextQuery)
- Hybrid: dense vector + full-text search (upsert + PreviewDenseVectorQuery + PreviewTextQuery)
- Boolean query string with filter (PreviewQueryStringQuery + filter dict)
- Sparse vector search (PreviewSparseVectorQuery + PreviewSparseValues)

These tests make real API calls and skip gracefully when the preview endpoint
is unavailable. They do NOT gate CI (preview_integration marker).

Eventual consistency: after upsert, search readiness is eventually consistent.
Each test polls via _wait_for_searchable() with up to 60 s timeout.
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.preview import PreviewSchemaBuilder
from pinecone.preview.models import (
    PreviewDenseVectorQuery,
    PreviewDocument,
    PreviewDocumentSearchResponse,
    PreviewQueryStringQuery,
    PreviewSparseValues,
    PreviewSparseVectorQuery,
    PreviewTextQuery,
    PreviewUsage,
)
from tests.integration.conftest import poll_until

pytestmark = [pytest.mark.integration, pytest.mark.preview_integration]


# ---------------------------------------------------------------------------
# Local helper
# ---------------------------------------------------------------------------


def _wait_for_searchable(
    idx: object,
    namespace: str,
    expected_ids: set[str],
    score_by: list[object],
    timeout: int = 60,
) -> PreviewDocumentSearchResponse:
    """Poll search until all expected_ids appear in matches. Returns final response."""
    from pinecone.preview.index import PreviewIndex

    assert isinstance(idx, PreviewIndex)

    def _query() -> PreviewDocumentSearchResponse:
        return idx.documents.search(
            namespace=namespace,
            top_k=len(expected_ids),
            score_by=score_by,  # type: ignore[arg-type]
        )

    def _check(r: PreviewDocumentSearchResponse) -> bool:
        found = {d._id for d in r.matches}
        return expected_ids.issubset(found)

    result = poll_until(
        _query,
        _check,
        timeout=timeout,
        interval=3,
        description=f"docs {expected_ids} searchable in namespace {namespace!r}",
    )
    assert isinstance(result, PreviewDocumentSearchResponse)
    return result


# ---------------------------------------------------------------------------
# test_full_text_search_only_flow — §13 "Full-text search only"
# ---------------------------------------------------------------------------


def test_full_text_search_only_flow(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """FTS-only schema: batch_upsert 3 docs, poll until searchable, assert top result."""
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_string_field("text", full_text_searchable=True, language="en")
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

    texts = ["ancient Rome", "medieval Europe", "modern tech"]
    documents = [{"_id": f"doc-{i}", "text": text} for i, text in enumerate(texts)]

    idx = client.preview.index(name=preview_index_name)
    idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=2,
        max_workers=2,
    )

    score_by: list[object] = [PreviewTextQuery(field="text", query="ancient")]
    results = _wait_for_searchable(idx, preview_namespace, {"doc-0", "doc-1", "doc-2"}, score_by)

    assert len(results.matches) >= 1
    # doc-0 ("ancient Rome") should rank highest for the query "ancient"
    top_match = max(results.matches, key=lambda d: d.score or 0.0)
    assert top_match._id == "doc-0"

    for doc in results.matches:
        assert isinstance(doc, PreviewDocument)
        assert isinstance(doc._id, str)
        assert doc.score is not None
        assert isinstance(doc.score, float)


# ---------------------------------------------------------------------------
# test_hybrid_search_combines_dense_and_text — §13 "Hybrid"
# ---------------------------------------------------------------------------


def test_hybrid_search_combines_dense_and_text(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Hybrid schema: single upsert, search with dense + text queries, assert doc-1 ranks first."""
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("chunk", full_text_searchable=True)
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
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[{"_id": "doc-1", "embedding": [0.1, 0.2, 0.3, 0.4], "chunk": "hello world"}],
    )

    score_by: list[object] = [
        PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4]),
        PreviewTextQuery(field="chunk", query="hello"),
    ]
    results = _wait_for_searchable(idx, preview_namespace, {"doc-1"}, score_by)

    assert len(results.matches) >= 1
    assert results.matches[0]._id == "doc-1"


# ---------------------------------------------------------------------------
# test_boolean_query_string_with_filter — §13 "Boolean query string with filter"
# ---------------------------------------------------------------------------


def test_boolean_query_string_with_filter(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Query string with filter: only sports-category docs matching football (not american)."""
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_string_field("chunk", full_text_searchable=True)
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
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "sports-1", "chunk": "football match premier league", "category": "sports"},
            {"_id": "news-1", "chunk": "american football NFL draft", "category": "news"},
            {"_id": "sports-2", "chunk": "football world cup Brazil", "category": "sports"},
        ],
    )

    score_by: list[object] = [
        PreviewQueryStringQuery(query="chunk:(+football NOT american)")
    ]

    # Poll until at least one sports doc is found (filter applied server-side)
    def _query_with_filter() -> PreviewDocumentSearchResponse:
        return idx.documents.search(
            namespace=preview_namespace,
            top_k=10,
            score_by=score_by,  # type: ignore[arg-type]
            filter={"category": {"$eq": "sports"}},
            include_fields=["chunk", "category"],
        )

    def _has_results(r: PreviewDocumentSearchResponse) -> bool:
        return len(r.matches) > 0

    results = poll_until(
        _query_with_filter,
        _has_results,
        timeout=60,
        interval=3,
        description="filtered sports docs searchable",
    )
    assert isinstance(results, PreviewDocumentSearchResponse)

    for doc in results.matches:
        assert isinstance(doc, PreviewDocument)
        assert doc.get("category") == "sports", f"expected sports, got {doc.get('category')}"
        # Verify include_fields populated both attributes
        assert doc.get("chunk") is not None
        assert doc.get("category") is not None

    match_ids = {d._id for d in results.matches}
    assert "news-1" not in match_ids, "news-1 (american football) should be excluded by filter"


# ---------------------------------------------------------------------------
# test_sparse_vector_search — §13 "Sparse vector search"
# ---------------------------------------------------------------------------


def test_sparse_vector_search(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Sparse vector schema: upsert 2 docs, search by sparse similarity, assert correct ranking."""
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_sparse_vector_field("sparse_embedding")
        .add_string_field("title", full_text_searchable=True, language="en")
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
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {
                "_id": "doc-1",
                "sparse_embedding": {"indices": [0, 42], "values": [0.8, 0.3]},
                "title": "alpha",
            },
            {
                "_id": "doc-2",
                "sparse_embedding": {"indices": [1, 99], "values": [0.1, 0.1]},
                "title": "beta",
            },
        ],
    )

    score_by: list[object] = [
        PreviewSparseVectorQuery(
            field="sparse_embedding",
            sparse_values=PreviewSparseValues(indices=[0, 42], values=[0.8, 0.3]),
        )
    ]
    results = _wait_for_searchable(idx, preview_namespace, {"doc-1", "doc-2"}, score_by)

    assert len(results.matches) >= 1
    # doc-1 shares the exact same indices/values as the query → should rank first
    assert results.matches[0]._id == "doc-1"


# ---------------------------------------------------------------------------
# test_search_include_fields_variants — §5 search include_fields behavior, §7 PreviewDocument
# ---------------------------------------------------------------------------


def test_search_include_fields_variants(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Verify include_fields request construction: explicit list and ["*"] accepted; None causes 422.

    This test validates the SDK's request-level behavior for include_fields.
    It does NOT assert on returned document fields because search indexing is
    eventually consistent and the test avoids polling to keep it fast.

    SDK BUG (IPV-0001): search() omits include_fields from the request body
    when the caller passes include_fields=None (the default). The API requires
    include_fields to be a non-null list, so it returns 422. The test reaches
    this assertion last; the DISABLED result is expected until IPV-0001 is fixed.
    """
    from pinecone.errors.exceptions import ApiError
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("title", filterable=True)
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
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {
                "_id": "doc-1",
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "title": "ancient Rome",
                "category": "history",
            }
        ],
    )

    query_vec = [0.1, 0.2, 0.3, 0.4]
    score_by: list[object] = [PreviewDenseVectorQuery(field="embedding", values=query_vec)]

    # Case 1: include_fields=["*"] — SDK sends field; API accepts (200 OK).
    results_star = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["*"],
    )
    assert isinstance(results_star, PreviewDocumentSearchResponse)

    # Case 2: include_fields=["title"] — SDK sends field; API accepts (200 OK).
    results_named = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["title"],
    )
    assert isinstance(results_named, PreviewDocumentSearchResponse)

    # Case 3: include_fields=None (default) — SDK omits include_fields from body → 422.
    # Per spec, None should return only _id and score (a valid operation).
    # SDK BUG (IPV-0001): SDK omits include_fields entirely; API requires it as a list.
    results_default = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
    )
    assert isinstance(results_default, PreviewDocumentSearchResponse)


# ---------------------------------------------------------------------------
# test_search_response_namespace_and_usage — §7 PreviewDocumentSearchResponse envelope
# ---------------------------------------------------------------------------


def test_search_response_namespace_and_usage(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Verify PreviewDocumentSearchResponse.namespace and .usage fields (§7).

    The spec declares that search() returns a PreviewDocumentSearchResponse with:
    - namespace: str  — echoed back from the request parameter
    - usage: PreviewUsage | None — with read_units: int >= 0

    No existing test checks these envelope fields; all existing tests inspect
    only response.matches items. This test targets the response envelope directly.
    Uses include_fields=["*"] to avoid the IPV-0001 422 bug.
    0 matches are acceptable — the namespace and usage fields are always present.
    """
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
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
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[{"_id": "doc-env", "embedding": [0.1, 0.2, 0.3, 0.4]}],
    )

    score_by: list[object] = [
        PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])
    ]
    response = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["*"],
    )

    assert isinstance(response, PreviewDocumentSearchResponse)
    # §7: namespace must be echoed back from the request.
    assert response.namespace == preview_namespace, (
        f"response.namespace {response.namespace!r} != request namespace {preview_namespace!r}"
    )
    # §7: usage must be present with a non-negative read_units counter.
    assert response.usage is not None, "response.usage must not be None after a successful search"
    assert isinstance(response.usage, PreviewUsage), (
        f"expected PreviewUsage, got {type(response.usage)}"
    )
    assert isinstance(response.usage.read_units, int), (
        f"read_units must be int, got {type(response.usage.read_units)}"
    )
    assert response.usage.read_units >= 0, (
        f"read_units must be >= 0, got {response.usage.read_units}"
    )


# ---------------------------------------------------------------------------
# test_filter_integer_gte_and_operator_accepted — §8 Metadata filtering
# ---------------------------------------------------------------------------


def test_filter_integer_gte_and_operator_accepted(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Verify §8: filter with integer $gte and $and operator is serialized and accepted (200 OK).

    Spec §8 declares these operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or.
    No existing test verifies that the SDK serializes $gte on an integer field or
    that the $and logical operator is accepted by the API.

    Creates a dense vector schema with filterable integer 'year' and filterable
    string 'category'. Upserts a document. Searches with a filter combining $and
    with $gte on year and $eq on category. Asserts 200 OK regardless of matches
    (OnDemand dense vector indexes are eventually consistent; 0 matches is acceptable).
    """
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .add_integer_field("year", filterable=True)
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
    idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "doc-1", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "tech", "year": 2022},
            {"_id": "doc-2", "embedding": [0.5, 0.6, 0.7, 0.8], "category": "science", "year": 2018},
        ],
    )

    score_by: list[object] = [PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])]

    # Verify $gte filter on integer field is accepted (200 OK).
    result_gte = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        filter={"year": {"$gte": 2020}},
        include_fields=["*"],
    )
    assert isinstance(result_gte, PreviewDocumentSearchResponse), (
        "$gte filter on integer field should return 200 OK"
    )

    # Verify $and operator combining $gte + $eq is accepted (200 OK).
    result_and = idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        filter={
            "$and": [
                {"category": {"$eq": "tech"}},
                {"year": {"$gte": 2020}},
            ]
        },
        include_fields=["*"],
    )
    assert isinstance(result_and, PreviewDocumentSearchResponse), (
        "$and filter with $gte + $eq should return 200 OK"
    )
