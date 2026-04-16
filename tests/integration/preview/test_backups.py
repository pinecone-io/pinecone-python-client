"""Integration tests for preview backup lifecycle (create, list).

Tests run against Pinecone API version 2026-01.alpha. Only ``create_backup``
and ``list_backups`` (scoped to an index) are in this API version's endpoint
set — standalone describe, delete, and restore operations are NOT tested here.

Tests skip gracefully when the preview endpoint is unavailable. They do NOT
gate CI (preview_integration mark).
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.preview import PreviewSchemaBuilder
from pinecone.preview.models import PreviewBackupModel
from tests.integration.conftest import poll_until

pytestmark = [pytest.mark.integration, pytest.mark.preview_integration]


def _simple_fts_schema() -> dict:  # type: ignore[type-arg]
    return PreviewSchemaBuilder().add_string_field("text", full_text_searchable=True).build()


def _is_ready(m: object) -> bool:
    from pinecone.preview.models import PreviewIndexModel

    return isinstance(m, PreviewIndexModel) and m.status.state == "Ready"


@pytest.fixture
def ready_preview_index(
    client: Pinecone,
    preview_index_name: str,
    cleanup_preview_indexes: list[str],
    require_preview: None,
) -> str:
    """Create a minimal FTS preview index, wait until Ready, return its name."""
    cleanup_preview_indexes.append(preview_index_name)
    client.preview.indexes.create(
        name=preview_index_name,
        schema=_simple_fts_schema(),
    )
    poll_until(
        lambda: client.preview.indexes.describe(preview_index_name),
        _is_ready,
        timeout=300,
        interval=5,
        description=f"preview index {preview_index_name} ready",
    )
    return preview_index_name


def test_create_backup_returns_initializing_backup(
    client: Pinecone,
    ready_preview_index: str,
) -> None:
    backup = client.preview.indexes.create_backup(
        ready_preview_index,
        name="before-migration",
        description="Snapshot before schema change",
    )
    assert isinstance(backup, PreviewBackupModel)
    assert isinstance(backup.backup_id, str) and len(backup.backup_id) > 0
    assert isinstance(backup.status, str) and len(backup.status) > 0


def test_list_backups_includes_created_backup(
    client: Pinecone,
    ready_preview_index: str,
) -> None:
    backup = client.preview.indexes.create_backup(
        ready_preview_index,
        name="integration-list-test",
    )
    backup_ids = [
        b.backup_id for b in client.preview.indexes.list_backups(ready_preview_index)
    ]
    assert backup.backup_id in backup_ids
    for item in client.preview.indexes.list_backups(ready_preview_index):
        assert isinstance(item, PreviewBackupModel)
        assert isinstance(item.backup_id, str) and len(item.backup_id) > 0
        assert isinstance(item.status, str)
        assert isinstance(item.created_at, str)


def test_list_backups_empty_for_new_index(
    client: Pinecone,
    ready_preview_index: str,
) -> None:
    items = list(client.preview.indexes.list_backups(ready_preview_index))
    assert len(items) == 0
