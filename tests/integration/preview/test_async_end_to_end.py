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
from pinecone.preview import SchemaBuilder
from pinecone.preview.models import (
    PreviewDocument,
    PreviewDocumentSearchResponse,
    PreviewIndexModel,
    PreviewTextQuery,
)
from pinecone.models.batch import BatchResult
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
