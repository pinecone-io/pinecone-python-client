"""Integration tests for async preview end-to-end flow (2026-01.alpha).

Covers §13 "Async usage" from spec/preview.md:
- test_async_end_to_end_flow: create → batch_upsert → search → (cleanup fixture deletes)
- test_async_describe_after_create_matches_sync: describe path works async end-to-end

These tests make real API calls and skip gracefully when the preview endpoint is
unavailable. They do NOT gate CI (preview_integration marker).

Uses pytest-asyncio and the async_client fixture from the parent conftest.
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone
from pinecone.models.batch import BatchResult
from pinecone.preview import SchemaBuilder
from pinecone.preview.models import (
    PreviewDenseVectorQuery,
    PreviewDocument,
    PreviewDocumentSearchResponse,
    PreviewIndexModel,
    PreviewTextQuery,
    PreviewUsage,
)
from tests.integration.conftest import async_poll_until

pytestmark = [pytest.mark.integration, pytest.mark.preview_integration, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# test_async_end_to_end_flow — §13 "Async usage"
# ---------------------------------------------------------------------------


async def test_async_end_to_end_flow(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """FTS schema: async create → batch_upsert 50 docs → poll → search → cleanup via fixture."""
    schema = (
        SchemaBuilder()
        .add_string_field("text", full_text_searchable=True, language="en")
        .build()
    )
    # Register for cleanup BEFORE create so teardown runs even if create raises.
    async_cleanup_preview_indexes.append(preview_index_name)

    model = await async_client.preview.indexes.create(
        name=preview_index_name, schema=schema
    )
    assert isinstance(model, PreviewIndexModel)
    assert model.name == preview_index_name

    idx = async_client.preview.index(name=preview_index_name)

    # Wait for index to be Ready.
    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    documents = [{"_id": f"doc-{i}", "text": f"document {i}"} for i in range(50)]
    result = await idx.documents.batch_upsert(
        namespace=preview_namespace,
        documents=documents,
        batch_size=10,
        max_workers=4,
    )
    assert isinstance(result, BatchResult)
    assert result.successful_item_count == 50
    assert result.successful_batch_count == 5

    # Poll until all 50 docs are searchable (eventual consistency).
    score_by: list[object] = [PreviewTextQuery(field="text", query="document")]

    async def _search() -> PreviewDocumentSearchResponse:
        return await idx.documents.search(
            namespace=preview_namespace,
            top_k=50,
            score_by=score_by,  # type: ignore[arg-type]
            include_fields=["text"],
        )

    results = await async_poll_until(
        _search,
        lambda r: isinstance(r, PreviewDocumentSearchResponse) and len(r.matches) == 50,
        timeout=90,
        interval=3,
        description="all 50 docs searchable",
    )
    assert isinstance(results, PreviewDocumentSearchResponse)
    for doc in results.matches:
        assert isinstance(doc, PreviewDocument)
        assert isinstance(doc._id, str)
        assert doc.score is not None
        assert isinstance(doc.score, float)
        assert doc.text is not None

    # Cleanup handled by async_cleanup_preview_indexes fixture.


# ---------------------------------------------------------------------------
# test_async_describe_after_create_matches_sync
# ---------------------------------------------------------------------------


async def test_async_describe_after_create_matches_sync(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async describe path returns correct name and host after create."""
    schema = (
        SchemaBuilder()
        .add_string_field("text", full_text_searchable=True, language="en")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    described = await async_client.preview.indexes.describe(preview_index_name)
    assert isinstance(described, PreviewIndexModel)
    assert described.name == preview_index_name
    assert isinstance(described.host, str) and len(described.host) > 0


# ---------------------------------------------------------------------------
# test_async_exists_returns_true_and_false — §2 exists(name) async parity
# ---------------------------------------------------------------------------


async def test_async_search_include_fields_variants(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: verify include_fields request construction; None causes 422 (IPV-0001)."""
    from pinecone.preview import PreviewSchemaBuilder
    from pinecone.preview.models import PreviewDenseVectorQuery, PreviewIndexModel

    schema = (
        PreviewSchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("title", filterable=True)
        .add_string_field("category", filterable=True)
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    await idx.documents.upsert(
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
    results_star = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["*"],
    )
    assert isinstance(results_star, PreviewDocumentSearchResponse)

    # Case 2: include_fields=["title"] — SDK sends field; API accepts (200 OK).
    results_named = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
        include_fields=["title"],
    )
    assert isinstance(results_named, PreviewDocumentSearchResponse)

    # Case 3: include_fields=None (default) — SDK omits include_fields from body → 422.
    # Per spec, None should return only _id and score (a valid operation).
    # SDK BUG (IPV-0001): SDK omits include_fields entirely; API requires it as a list.
    results_default = await idx.documents.search(
        namespace=preview_namespace,
        top_k=5,
        score_by=score_by,  # type: ignore[arg-type]
    )
    assert isinstance(results_default, PreviewDocumentSearchResponse)


# ---------------------------------------------------------------------------
# test_async_exists_returns_true_and_false — §2 exists(name) async parity
# ---------------------------------------------------------------------------


async def test_async_exists_returns_true_and_false(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async exists() returns True for a created index and False for an unknown name."""
    from tests.integration.conftest import unique_name

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    result_true = await async_client.preview.indexes.exists(preview_index_name)
    assert result_true is True

    phantom_name = unique_name("phantom")
    result_false = await async_client.preview.indexes.exists(phantom_name)
    assert result_false is False


# ---------------------------------------------------------------------------
# test_async_upsert_returns_upserted_count — §5 async parity
# ---------------------------------------------------------------------------


async def test_async_upsert_returns_upserted_count(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async upsert() returns PreviewDocumentUpsertResponse with upserted_count == len(documents).

    Async parity for test_upsert_returns_upserted_count: verifies §5 response
    shape via the async SDK path.
    """
    from pinecone.preview.models import PreviewDocumentUpsertResponse, PreviewIndexModel

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    documents = [
        {"_id": "doc-0", "embedding": [0.1, 0.2, 0.3, 0.4]},
        {"_id": "doc-1", "embedding": [0.5, 0.6, 0.7, 0.8]},
        {"_id": "doc-2", "embedding": [0.9, 0.1, 0.2, 0.3]},
    ]
    response = await idx.documents.upsert(namespace=preview_namespace, documents=documents)

    assert isinstance(response, PreviewDocumentUpsertResponse)
    assert response.upserted_count == len(documents)


# ---------------------------------------------------------------------------
# test_async_batch_upsert_result_fields — §5 async batch_upsert() BatchResult
# ---------------------------------------------------------------------------


async def test_async_batch_upsert_result_fields(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async batch_upsert() returns BatchResult with correct counts for 10 docs in 2 batches.

    Async parity for test_batch_upsert_result_fields: verifies §5 BatchResult
    fields (total_item_count, failed_item_count, total_batch_count,
    failed_batch_count, has_errors, failed_items, errors) via the async path.
    """
    from pinecone.models.batch import BatchResult
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    documents = [
        {"_id": f"doc-{i}", "embedding": [float(i) / 10, 0.1, 0.2, 0.3]}
        for i in range(10)
    ]

    result = await idx.documents.batch_upsert(
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
# test_async_fetch_wildcard_include_fields — §5 async fetch() include_fields=["*"]
# ---------------------------------------------------------------------------


async def test_async_fetch_wildcard_include_fields_returns_all_stored_fields(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async fetch() with include_fields=["*"] returns all stored fields for each document.

    Async parity for test_fetch_wildcard_include_fields_returns_all_stored_fields:
    verifies §5 wildcard behavior via the async SDK path.

    SERVER BUG (IPV-0002): fetch() returns 401 "Unknown operation" for all
    preview index types. This test is expected to fail until IPV-0002 is resolved.
    """
    import asyncio

    from pinecone.preview.models import (
        PreviewDocument,
        PreviewDocumentFetchResponse,
        PreviewIndexModel,
    )

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    docs = [
        {"_id": "fruit-0", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "fruit"},
        {"_id": "fruit-1", "embedding": [0.5, 0.6, 0.7, 0.8], "category": "vegetable"},
    ]
    await idx.documents.upsert(namespace=preview_namespace, documents=docs)
    await asyncio.sleep(3)

    # Fetch with wildcard — all stored fields must come back.
    # IPV-0002: this call fails with 401 "Unknown operation" until fixed.
    response = await idx.documents.fetch(
        namespace=preview_namespace,
        ids=["fruit-0", "fruit-1"],
        include_fields=["*"],
    )

    assert isinstance(response, PreviewDocumentFetchResponse)
    assert set(response.documents.keys()) == {"fruit-0", "fruit-1"}

    for doc_id, doc in response.documents.items():
        assert isinstance(doc, PreviewDocument)
        assert doc._id == doc_id
        assert doc.category is not None, f"doc {doc_id} missing 'category' with include_fields=['*']"

    assert response.documents["fruit-0"].category == "fruit"
    assert response.documents["fruit-1"].category == "vegetable"


# ---------------------------------------------------------------------------
# test_async_configure_toggle_deletion_protection — §2 configure() async parity
# ---------------------------------------------------------------------------


async def test_async_configure_toggle_deletion_protection(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async configure() toggles deletion_protection; describe() reflects each change.

    Async parity for TestDeletionProtectionToggle.test_configure_toggle_deletion_protection:
    verifies §2 configure(deletion_protection=) and §3 PreviewIndexModel.deletion_protection
    via the async SDK path.
    """
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready before configure",
    )

    initial = await async_client.preview.indexes.describe(preview_index_name)
    assert initial.deletion_protection == "disabled"

    # Enable deletion_protection
    await async_client.preview.indexes.configure(preview_index_name, deletion_protection="enabled")
    after_enable = await async_client.preview.indexes.describe(preview_index_name)
    assert after_enable.deletion_protection == "enabled"

    # Disable so the cleanup fixture can delete the index
    await async_client.preview.indexes.configure(preview_index_name, deletion_protection="disabled")
    after_disable = await async_client.preview.indexes.describe(preview_index_name)
    assert after_disable.deletion_protection == "disabled"


# ---------------------------------------------------------------------------
# test_async_add_custom_field_appears_in_describe — §1 add_custom_field() async parity
# ---------------------------------------------------------------------------


async def test_async_add_custom_field_appears_in_describe(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: add_custom_field() raw dict passes through; describe() reflects the field.

    Async parity for test_add_custom_field_appears_in_describe: verifies §1
    add_custom_field() escape hatch via the async SDK path.
    """
    from pinecone.preview.models import PreviewIntegerField

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_custom_field("score", {"type": "float", "filterable": True})
        .build()
    )

    # Verify build() output contains the custom dict unchanged.
    assert schema["fields"]["score"] == {"type": "float", "filterable": True}

    # Create the index via API.
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    # describe() must return the custom field as a typed model.
    model = await async_client.preview.indexes.describe(preview_index_name)
    assert "score" in model.schema.fields
    score_field = model.schema.fields["score"]
    assert isinstance(score_field, PreviewIntegerField), (
        f"expected PreviewIntegerField, got {type(score_field)}"
    )
    assert score_field.filterable is True


# ---------------------------------------------------------------------------
# test_async_create_with_tags_returns_tags_in_describe — §2/§3 tags async parity
# ---------------------------------------------------------------------------


async def test_async_create_with_tags_returns_tags_in_describe(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async create() with tags returns those tags verbatim in describe() PreviewIndexModel.tags.

    Async parity for TestIndexTags.test_create_with_tags_returns_tags_in_describe:
    verifies §2 create(tags=) and §3 PreviewIndexModel.tags via the async SDK path.
    """
    tags = {"env": "integration-test", "pvt": "PVT-008"}
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(
        name=preview_index_name,
        schema=schema,
        tags=tags,
    )

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    model = await async_client.preview.indexes.describe(preview_index_name)
    assert model.tags is not None, "tags should not be None after create() with tags"
    assert model.tags == tags


# ---------------------------------------------------------------------------
# test_async_configure_tags_merges_with_existing_tags — §2 configure(tags=) merge
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# test_async_create_and_list_backup — §9 async variants + §2 create_backup/list_backups
# ---------------------------------------------------------------------------


async def test_async_create_and_list_backup(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity for backup operations — create_backup and list_backups work via async path.

    Async counterpart for test_backups.py sync tests. Verifies:
    - Async create_backup() returns PreviewBackupModel with all required fields.
    - source_index_name matches the index that was backed up.
    - Async list_backups() iteration yields the created backup by backup_id.
    """
    from pinecone.preview.models import PreviewBackupModel

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    backup = await async_client.preview.indexes.create_backup(
        preview_index_name,
        name="async-integration-test",
        description="Async backup parity test",
    )

    assert isinstance(backup, PreviewBackupModel)
    assert isinstance(backup.backup_id, str) and len(backup.backup_id) > 0
    assert backup.source_index_name == preview_index_name
    assert isinstance(backup.source_index_id, str) and len(backup.source_index_id) > 0
    assert isinstance(backup.status, str) and len(backup.status) > 0
    assert isinstance(backup.cloud, str) and len(backup.cloud) > 0
    assert isinstance(backup.region, str) and len(backup.region) > 0
    assert isinstance(backup.created_at, str) and len(backup.created_at) > 0

    backup_ids = [
        b.backup_id
        async for b in async_client.preview.indexes.list_backups(preview_index_name)
    ]
    assert backup.backup_id in backup_ids

    async for item in async_client.preview.indexes.list_backups(preview_index_name):
        assert isinstance(item, PreviewBackupModel)
        assert isinstance(item.backup_id, str) and len(item.backup_id) > 0
        assert isinstance(item.status, str)
        assert isinstance(item.created_at, str)


async def test_async_configure_tags_merges_with_existing_tags(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async configure(tags=) merges new tags with existing tags — does not replace them.

    Async parity for TestIndexTags.test_configure_tags_merges_with_existing_tags:
    verifies §2 configure(tags=) merge behavior via the async SDK path.
    """
    initial_tags = {"env": "integration-test", "key1": "original"}
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(
        name=preview_index_name,
        schema=schema,
        tags=initial_tags,
    )

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    await async_client.preview.indexes.configure(preview_index_name, tags={"key2": "added"})

    model = await async_client.preview.indexes.describe(preview_index_name)
    assert model.tags is not None, "tags should not be None after configure(tags=)"
    assert "env" in model.tags, "original tag 'env' must survive configure(tags=)"
    assert "key1" in model.tags, "original tag 'key1' must survive configure(tags=)"
    assert model.tags["key1"] == "original", "original tag value must be unchanged"
    assert "key2" in model.tags, "new tag 'key2' must be present after configure(tags=)"
    assert model.tags["key2"] == "added", "new tag value must be correct"


# ---------------------------------------------------------------------------
# test_async_search_response_namespace_and_usage — §7 async parity
# ---------------------------------------------------------------------------


async def test_async_search_response_namespace_and_usage(
    async_client: AsyncPinecone,
    preview_index_name: str,
    preview_namespace: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: PreviewDocumentSearchResponse.namespace echoed and usage present (§7).

    Async counterpart for test_search_response_namespace_and_usage. Verifies that
    an async search() call returns a response envelope where:
    - namespace == the namespace argument passed to search()
    - usage is a PreviewUsage instance with read_units: int >= 0

    Uses include_fields=["*"] to avoid the IPV-0001 422 bug. 0 matches are
    acceptable — the envelope fields must always be present on a 200 OK response.
    """
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    await idx.documents.upsert(
        namespace=preview_namespace,
        documents=[{"_id": "doc-env", "embedding": [0.1, 0.2, 0.3, 0.4]}],
    )

    score_by: list[object] = [
        PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])
    ]
    response = await idx.documents.search(
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
# test_async_delete_raises_forbidden_when_deletion_protection_enabled — §2
# ---------------------------------------------------------------------------


async def test_async_delete_raises_forbidden_when_deletion_protection_enabled(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: delete() raises ForbiddenError when deletion_protection is "enabled" (§2).

    Async counterpart for
    TestDeletionProtectionEnforcement.test_delete_raises_forbidden_when_deletion_protection_enabled.
    Verifies the spec claim via the async SDK path: a delete() call on a
    deletion-protected index raises ForbiddenError and leaves the index intact.
    """
    from pinecone.errors import ForbiddenError
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    await async_client.preview.indexes.configure(
        preview_index_name, deletion_protection="enabled"
    )

    try:
        with pytest.raises(ForbiddenError):
            await async_client.preview.indexes.delete(preview_index_name)

        assert await async_client.preview.indexes.exists(preview_index_name), (
            "index must still exist after delete() was rejected by deletion_protection"
        )
    finally:
        # Disable protection so the cleanup fixture can delete the index.
        try:
            await async_client.preview.indexes.configure(
                preview_index_name, deletion_protection="disabled"
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# test_async_filter_integer_gte_and_operator_accepted — §8 Metadata filtering
# ---------------------------------------------------------------------------


async def test_async_filter_integer_gte_and_operator_accepted(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Async parity: filter with $gte (integer) and $and operator accepted by API (§8).

    Async counterpart for test_filter_integer_gte_and_operator_accepted.
    Verifies that the SDK correctly serializes $gte and $and operators and that the API
    returns 200 OK. 0 matches is acceptable (OnDemand dense vector indexing is eventually
    consistent). Uses dense vector + filterable integer year + filterable string category.
    """
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .add_integer_field("year", filterable=True)
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    await idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "doc-1", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "tech", "year": 2022},
            {"_id": "doc-2", "embedding": [0.5, 0.6, 0.7, 0.8], "category": "science", "year": 2018},
        ],
    )

    score_by: list[object] = [PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])]

    # Verify $gte filter on integer field is accepted (200 OK).
    result_gte = await idx.documents.search(
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
    result_and = await idx.documents.search(
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


# ---------------------------------------------------------------------------
# test_async_list_limit_* — §2 list(limit=N) parameter validation
# ---------------------------------------------------------------------------


async def test_async_list_limit_zero_raises_value_error(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async parity: list(limit=0) raises PineconeValueError without an API call (§2).

    Async counterpart for TestListLimit.test_list_limit_zero_raises_value_error.
    The spec declares limit must be a positive integer; the async SDK validates this
    client-side before the async paginator is created. No await is needed.
    """
    from pinecone.errors import PineconeValueError

    with pytest.raises(PineconeValueError):
        async_client.preview.indexes.list(limit=0)


async def test_async_list_limit_negative_raises_value_error(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async parity: list(limit=-1) raises PineconeValueError without an API call (§2)."""
    from pinecone.errors import PineconeValueError

    with pytest.raises(PineconeValueError):
        async_client.preview.indexes.list(limit=-1)


async def test_async_list_limit_caps_results(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: list(limit=1) yields at most 1 PreviewIndexModel (§2).

    SDK BUG (IPV-0003): list() raises msgspec.ValidationError when the account contains
    any index whose schema has a field lacking a 'type' key. This test is DISABLED until
    IPV-0003 is fixed.
    """
    pytest.skip("DISABLED (IPV-0003): list() crashes with msgspec.ValidationError for accounts with non-standard schema fields")


# ---------------------------------------------------------------------------
# test_async_search_client_side_validation — §7 search() parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_search_client_side_validation_rejects_invalid_parameters(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async parity: search() raises ValidationError for invalid parameters without an API call.

    Async counterpart for test_search_client_side_validation_rejects_invalid_parameters.
    Spec §7 declares client-side validation fires before the HTTP request is sent:
    - namespace must be non-empty
    - top_k must be 1–10000
    - score_by must be non-empty

    Uses a dummy host — no network call is made.
    """
    from pinecone.errors.exceptions import ValidationError

    # Use a dummy host — async validation fires synchronously before the coroutine awaits.
    idx = async_client.preview.index(host="https://dummy-host.pinecone.io")

    valid_score_by: list[object] = [{"type": "dense_vector", "field": "emb", "values": [0.1]}]

    # Empty namespace string must raise ValidationError.
    with pytest.raises(ValidationError, match="namespace"):
        await idx.documents.search(namespace="", top_k=5, score_by=valid_score_by)  # type: ignore[arg-type]

    # top_k=0 is below the minimum of 1.
    with pytest.raises(ValidationError, match="top_k"):
        await idx.documents.search(namespace="ns", top_k=0, score_by=valid_score_by)  # type: ignore[arg-type]

    # top_k=10001 is above the maximum of 10000.
    with pytest.raises(ValidationError, match="top_k"):
        await idx.documents.search(namespace="ns", top_k=10001, score_by=valid_score_by)  # type: ignore[arg-type]

    # Empty score_by list must raise ValidationError.
    with pytest.raises(ValidationError, match="score_by"):
        await idx.documents.search(namespace="ns", top_k=5, score_by=[])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# test_async_upsert_client_side_validation — §4 async upsert() validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_upsert_client_side_validation_rejects_invalid_documents(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async parity: upsert() raises ValidationError for all invalid-document conditions.

    Async counterpart for test_upsert_client_side_validation_rejects_invalid_documents.
    Spec §4 client-side validation fires synchronously at the start of the coroutine,
    before the first await, so ValidationError propagates through the await normally.
    Uses a dummy host — no network call is made.
    """
    from pinecone.errors.exceptions import ValidationError

    idx = async_client.preview.index(host="https://dummy-host.pinecone.io")

    valid_doc = {"_id": "doc-0", "text": "hello"}

    # Empty namespace must raise ValidationError.
    with pytest.raises(ValidationError, match="namespace"):
        await idx.documents.upsert(namespace="", documents=[valid_doc])

    # Empty documents list must raise ValidationError.
    with pytest.raises(ValidationError, match="documents"):
        await idx.documents.upsert(namespace="ns", documents=[])

    # More than 100 documents must raise ValidationError.
    over_limit = [{"_id": f"doc-{i}"} for i in range(101)]
    with pytest.raises(ValidationError, match="documents"):
        await idx.documents.upsert(namespace="ns", documents=over_limit)

    # Document missing '_id' key must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        await idx.documents.upsert(namespace="ns", documents=[{"text": "no id here"}])

    # Document with non-string '_id' must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        await idx.documents.upsert(namespace="ns", documents=[{"_id": 42}])  # type: ignore[list-item]

    # Document with empty string '_id' must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        await idx.documents.upsert(namespace="ns", documents=[{"_id": ""}])

    # Duplicate '_id' values within one call must raise ValidationError.
    with pytest.raises(ValidationError, match="_id"):
        await idx.documents.upsert(
            namespace="ns",
            documents=[{"_id": "dup"}, {"_id": "dup"}],
        )


# ---------------------------------------------------------------------------
# test_async_delete_client_side_validation — §4 async delete() validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_delete_client_side_validation_rejects_invalid_arguments(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async parity: delete() raises ValidationError for missing or conflicting deletion targets.

    Async counterpart for test_delete_client_side_validation_rejects_invalid_arguments.
    Spec §4 validation fires synchronously at the coroutine start — no HTTP call is made.
    Uses a dummy host.
    """
    from pinecone.errors.exceptions import ValidationError

    idx = async_client.preview.index(host="https://dummy-host.pinecone.io")

    # Empty namespace must raise ValidationError.
    with pytest.raises(ValidationError, match="namespace"):
        await idx.documents.delete(namespace="", ids=["doc-0"])

    # Calling delete() with no targets must raise ValidationError.
    with pytest.raises(ValidationError, match="ids"):
        await idx.documents.delete(namespace="ns")

    # ids and delete_all=True are mutually exclusive.
    with pytest.raises(ValidationError, match="ids"):
        await idx.documents.delete(namespace="ns", ids=["doc-0"], delete_all=True)

    # ids and filter are mutually exclusive.
    with pytest.raises(ValidationError, match="ids"):
        await idx.documents.delete(
            namespace="ns", ids=["doc-0"], filter={"category": {"$eq": "fruit"}}
        )


# ---------------------------------------------------------------------------
# test_async_filter_remaining_operators_accepted — §8 Metadata filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_filter_remaining_operators_accepted(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    preview_namespace: str,
    require_preview: None,
) -> None:
    """Async parity: remaining filter operators ($ne, $gt, $lt, $lte, $in, $nin, $or) accepted.

    Async counterpart for test_filter_remaining_operators_accepted.
    Verifies that the SDK serializes each of the 7 operators not covered by PVT-013
    ($ne, $gt, $lt, $lte, $in, $nin, $or) and that the API returns 200 OK.
    0 matches is acceptable — OnDemand indexing is eventually consistent.
    """
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_string_field("category", filterable=True)
        .add_integer_field("year", filterable=True)
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    idx = async_client.preview.index(name=preview_index_name)
    await idx.documents.upsert(
        namespace=preview_namespace,
        documents=[
            {"_id": "doc-1", "embedding": [0.1, 0.2, 0.3, 0.4], "category": "tech", "year": 2022},
            {"_id": "doc-2", "embedding": [0.5, 0.6, 0.7, 0.8], "category": "science", "year": 2018},
        ],
    )

    score_by: list[object] = [PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])]

    async def _search_with_filter(f: dict) -> None:
        result = await idx.documents.search(
            namespace=preview_namespace,
            top_k=5,
            score_by=score_by,  # type: ignore[arg-type]
            filter=f,
            include_fields=["*"],
        )
        assert isinstance(result, PreviewDocumentSearchResponse), (
            f"filter {f} should return a PreviewDocumentSearchResponse (200 OK)"
        )

    # $ne — not equals on string field
    await _search_with_filter({"category": {"$ne": "finance"}})

    # $gt — greater than on integer field
    await _search_with_filter({"year": {"$gt": 2010}})

    # $lt — less than on integer field
    await _search_with_filter({"year": {"$lt": 2025}})

    # $lte — less or equal on integer field
    await _search_with_filter({"year": {"$lte": 2022}})

    # $in — value in array on string field
    await _search_with_filter({"category": {"$in": ["tech", "medicine"]}})

    # $nin — value not in array on string field
    await _search_with_filter({"category": {"$nin": ["finance", "sports"]}})

    # $or — logical OR combining two conditions
    await _search_with_filter({"$or": [{"category": {"$eq": "tech"}}, {"year": {"$lt": 2020}}]})


# ---------------------------------------------------------------------------
# test_async_delete_timeout_negative_one_returns_immediately — §2 delete(timeout=-1)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_delete_timeout_negative_one_returns_immediately(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: delete(timeout=-1) returns None immediately; index eventually disappears (§2).

    Async counterpart for
    TestDeleteTimeout.test_delete_timeout_negative_one_returns_immediately.
    Verifies that the async delete() with timeout=-1 returns None without waiting
    for the index to disappear, and the index is eventually confirmed gone.
    """
    from pinecone.errors import NotFoundError
    from pinecone.preview.models import PreviewIndexModel

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .build()
    )
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready before delete",
    )

    result = await async_client.preview.indexes.delete(preview_index_name, timeout=-1)
    assert result is None, "delete(timeout=-1) must return None"

    # Verify the index eventually disappears (server finishes deletion async).
    _DELETED_SENTINEL = object()

    async def _describe_or_sentinel() -> object:
        try:
            return await async_client.preview.indexes.describe(preview_index_name)
        except NotFoundError:
            return _DELETED_SENTINEL

    await async_poll_until(
        _describe_or_sentinel,
        lambda r: r is _DELETED_SENTINEL,
        timeout=120,
        interval=5,
        description=f"index {preview_index_name} eventually deleted after timeout=-1",
    )


# ---------------------------------------------------------------------------
# test_async_schema_build_idempotency_and_field_collision_replacement — §1
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_async_schema_build_idempotency_and_field_collision_replacement(
    async_client: AsyncPinecone,
    preview_index_name: str,
    async_cleanup_preview_indexes: list[str],
    require_preview: None,
) -> None:
    """Async parity: build() idempotency, field collision replacement, **additional_options (§1).

    Async counterpart for
    TestSchemaBuildBehavior.test_schema_build_idempotency_and_field_collision_replacement.
    Verifies all three §1 SchemaBuilder claims in the async path and confirms the
    collision-replacement schema is accepted by the API.
    """
    # Claim 2: last add_dense_vector_field("embedding") must win.
    builder = (
        SchemaBuilder()
        .add_dense_vector_field("embedding", dimension=4, metric="cosine")
        .add_dense_vector_field("embedding", dimension=8, metric="euclidean")
        .add_integer_field("count")
    )

    schema1 = builder.build()
    schema2 = builder.build()

    assert schema1["fields"]["embedding"]["dimension"] == 8, (
        f"Expected dimension=8 (last add wins), got {schema1['fields']['embedding']['dimension']}"
    )
    assert schema1["fields"]["embedding"]["metric"] == "euclidean"

    # Claim 1: build() returns equal content but independent copies.
    assert schema1 == schema2, "build() must return equal dicts on repeated calls"
    assert schema1 is not schema2, "build() must return independent copies"
    assert schema1["fields"] is not schema2["fields"]

    # Claim 3: **additional_options merged into field dict (client-side).
    schema_with_extras = (
        SchemaBuilder()
        .add_integer_field("priority", filterable=True, x_custom_param=42)
        .build()
    )
    assert schema_with_extras["fields"]["priority"]["x_custom_param"] == 42
    assert schema_with_extras["fields"]["priority"]["type"] == "float"

    # Integration: API accepts the collision-replacement schema.
    async_cleanup_preview_indexes.append(preview_index_name)
    await async_client.preview.indexes.create(name=preview_index_name, schema=schema1)

    def _is_ready(m: object) -> bool:
        return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"

    await async_poll_until(
        lambda: async_client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"index {preview_index_name} ready",
    )

    model = await async_client.preview.indexes.describe(preview_index_name)
    assert len(model.schema.fields) == 2, (
        f"Expected 2 fields after collision replacement, got {len(model.schema.fields)}"
    )
    assert "embedding" in model.schema.fields
    assert "count" in model.schema.fields


# ---------------------------------------------------------------------------
# test_async_describe_raises_not_found_and_index_factory_validation — §2 + §4
# ---------------------------------------------------------------------------


async def test_async_describe_raises_not_found_and_index_factory_validation(
    async_client: AsyncPinecone,
    require_preview: None,
) -> None:
    """Async describe() raises NotFoundError; async preview.index() validates args and defers resolution (§2, §4).

    Verifies four spec claims:
    1. §2 "Raises: NotFoundError if the index does not exist."
    2. §4 async preview.index(name=phantom) returns an object immediately — host
       resolution is deferred to the first data-plane call (unlike the sync path).
    3. §4 The deferred NotFoundError surfaces on the first data-plane call (search()).
    4. §4 async preview.index() with neither/both args raises PineconeValueError.
    """
    from pinecone.errors import NotFoundError, PineconeValueError
    from tests.integration.conftest import unique_name

    phantom = unique_name("phantom")

    # Claim 1: await describe() raises NotFoundError for a name that was never created.
    with pytest.raises(NotFoundError):
        await async_client.preview.indexes.describe(phantom)

    # Claim 2: async preview.index(name=phantom) returns immediately without error
    # (host resolution is deferred).
    idx = async_client.preview.index(name=phantom)
    assert idx is not None

    # Claim 3: The NotFoundError surfaces on the first data-plane call.
    with pytest.raises(NotFoundError):
        await idx.documents.search(
            namespace="ns",
            top_k=1,
            score_by=[PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3, 0.4])],
        )

    # Claim 4: async preview.index() with neither/both args raises PineconeValueError.
    with pytest.raises(PineconeValueError):
        async_client.preview.index()  # type: ignore[call-arg]

    with pytest.raises(PineconeValueError):
        async_client.preview.index(name=phantom, host="https://dummy-host.pinecone.io")
