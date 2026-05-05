"""Unit tests for Pinecone.create_index_from_backup."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._client import Pinecone
from pinecone.errors.exceptions import ValidationError
from pinecone.models.backups.model import CreateIndexFromBackupResponse
from pinecone.models.indexes.index import IndexModel
from tests.factories import make_index_response

BASE_URL = "https://api.pinecone.io"


@pytest.fixture
def pc() -> Pinecone:
    return Pinecone(api_key="test-key")


# ---------------------------------------------------------------------------
# Basic success
# ---------------------------------------------------------------------------


@respx.mock
def test_create_index_from_backup_basic(pc: Pinecone) -> None:
    """POST creates the index, then polling returns a ready IndexModel."""
    respx.post(f"{BASE_URL}/backups/bk-123/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-1", "index_id": "idx-1"},
        ),
    )
    respx.get(f"{BASE_URL}/indexes/restored-index").mock(
        return_value=httpx.Response(200, json=make_index_response(name="restored-index")),
    )

    result = pc.create_index_from_backup(name="restored-index", backup_id="bk-123")

    assert isinstance(result, IndexModel)
    assert result.name == "restored-index"


# ---------------------------------------------------------------------------
# Tags and deletion protection
# ---------------------------------------------------------------------------


@respx.mock
def test_create_index_from_backup_with_tags_and_protection(pc: Pinecone) -> None:
    """Tags and deletion_protection appear in the request body."""
    route = respx.post(f"{BASE_URL}/backups/bk-456/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-2", "index_id": "idx-2"},
        ),
    )
    respx.get(f"{BASE_URL}/indexes/my-restored").mock(
        return_value=httpx.Response(200, json=make_index_response(name="my-restored")),
    )

    pc.create_index_from_backup(
        name="my-restored",
        backup_id="bk-456",
        deletion_protection="enabled",
        tags={"env": "prod"},
    )

    request = route.calls[0].request
    import orjson

    body = orjson.loads(request.content)
    assert body["name"] == "my-restored"
    assert body["deletion_protection"] == "enabled"
    assert body["tags"] == {"env": "prod"}


# ---------------------------------------------------------------------------
# No-poll (timeout=-1)
# ---------------------------------------------------------------------------


@respx.mock
def test_create_index_from_backup_no_poll(pc: Pinecone) -> None:
    """When timeout=-1, returns CreateIndexFromBackupResponse immediately without polling."""
    respx.post(f"{BASE_URL}/backups/bk-789/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-3", "index_id": "idx-3"},
        ),
    )

    result = pc.create_index_from_backup(name="quick-restore", backup_id="bk-789", timeout=-1)

    assert isinstance(result, CreateIndexFromBackupResponse)
    assert result.restore_job_id == "rj-3"
    assert result.index_id == "idx-3"


@respx.mock
def test_create_index_from_backup_no_wait_returns_restore_job_id(pc: Pinecone) -> None:
    """timeout=-1 gives callers access to restore_job_id without polling."""
    respx.post(f"{BASE_URL}/backups/bk-nwt/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-nowait", "index_id": "idx-nowait"},
        ),
    )

    result = pc.create_index_from_backup(
        name="test-restore-nowait",
        backup_id="bk-nwt",
        timeout=-1,
    )

    assert isinstance(result, CreateIndexFromBackupResponse)
    assert result.restore_job_id
    assert result.index_id


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_create_index_from_backup_empty_name_raises(pc: Pinecone) -> None:
    with pytest.raises(ValidationError) as exc_info:
        pc.create_index_from_backup(name="", backup_id="bk-123")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


def test_create_index_from_backup_empty_backup_id_raises(pc: Pinecone) -> None:
    with pytest.raises(ValidationError) as exc_info:
        pc.create_index_from_backup(name="my-index", backup_id="")

    assert "backup_id" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


@patch("pinecone._internal.indexes_helpers.time.sleep")
@respx.mock
def test_create_index_from_backup_polls_until_ready(mock_sleep: object, pc: Pinecone) -> None:
    """Describe is called multiple times until the index becomes ready."""
    respx.post(f"{BASE_URL}/backups/bk-poll/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-4", "index_id": "idx-4"},
        ),
    )
    not_ready = make_index_response(
        name="poll-index",
        status={"ready": False, "state": "Initializing"},
    )
    ready = make_index_response(
        name="poll-index",
        status={"ready": True, "state": "Ready"},
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/poll-index").mock(
        side_effect=[
            httpx.Response(200, json=not_ready),
            httpx.Response(200, json=ready),
        ]
    )

    result = pc.create_index_from_backup(name="poll-index", backup_id="bk-poll", timeout=60)

    assert isinstance(result, IndexModel)
    assert result.name == "poll-index"
    # Describe should be called at least 2 times (first not-ready, then ready)
    assert describe_route.call_count >= 2
