"""Unit tests for Projects._cleanup_project_resources()."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ADMIN_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.admin.projects import Projects
from pinecone.errors.exceptions import NotFoundError
from pinecone.models.backups.list import BackupList
from pinecone.models.vectors.responses import Pagination

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ADMIN_API_VERSION)


@pytest.fixture
def projects(http_client: HTTPClient) -> Projects:
    return Projects(http=http_client)


def _make_index(name: str) -> MagicMock:
    idx = MagicMock()
    idx.name = name
    return idx


def _make_collection(name: str) -> MagicMock:
    col = MagicMock()
    col.name = name
    return col


def _make_backup(backup_id: str) -> MagicMock:
    bk = MagicMock()
    bk.backup_id = backup_id
    return bk


def _backup_page(*backup_ids: str, next_token: str | None = None) -> BackupList:
    """One page of backups, in the shape ``pc.backups.list()`` returns it.

    That call hands back one page plus a token to follow, so a plain list
    here would let a loop that reads only the first page pass.
    """
    return BackupList(
        [_make_backup(b) for b in backup_ids],  # type: ignore[misc]
        pagination=Pagination(next=next_token) if next_token is not None else None,
    )


def _make_assistant(name: str) -> MagicMock:
    a = MagicMock()
    a.name = name
    return a


def _empty_project(mock_pc: MagicMock) -> None:
    """Give every resource namespace an explicit empty list.

    A bare MagicMock iterates empty, so without this a loop the cleanup
    forgot to run and a loop over nothing look identical.
    """
    mock_pc.indexes.list.return_value = []
    mock_pc.collections.list.return_value = []
    mock_pc.assistants.list.return_value = []
    mock_pc.backups.list.return_value = _backup_page()


@patch("pinecone._client.Pinecone")
def test_cleanup_deletes_indexes(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify delete is called for each index returned by list."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.indexes.list.return_value = [_make_index("index-1"), _make_index("index-2")]

    projects._cleanup_project_resources(api_key="test-project-key")

    mock_pinecone_cls.assert_called_once_with(api_key="test-project-key")
    assert mock_pc.indexes.delete.call_count == 2
    mock_pc.indexes.delete.assert_any_call("index-1")
    mock_pc.indexes.delete.assert_any_call("index-2")
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_deletes_backups(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify delete is called for each backup returned by list."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.backups.list.return_value = _backup_page("bk-001", "bk-002")

    projects._cleanup_project_resources(api_key="test-project-key")

    assert mock_pc.backups.delete.call_count == 2
    mock_pc.backups.delete.assert_any_call(backup_id="bk-001")
    mock_pc.backups.delete.assert_any_call(backup_id="bk-002")
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_deletes_backups_on_every_page(
    mock_pinecone_cls: MagicMock, projects: Projects
) -> None:
    """Backups past the first page are deleted too.

    A project with more backups than fit one page used to keep the ones on
    later pages, which then blocked the project delete that follows cleanup.
    """
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.backups.list.side_effect = [
        _backup_page("bk-001", "bk-002", next_token="page-2"),
        _backup_page("bk-003", "bk-004"),
    ]

    projects._cleanup_project_resources(api_key="test-project-key")

    deleted = {kwargs["backup_id"] for _, kwargs in mock_pc.backups.delete.call_args_list}
    assert deleted == {"bk-001", "bk-002", "bk-003", "bk-004"}
    assert mock_pc.backups.list.call_args_list[-1] == call(pagination_token="page-2")


@patch("pinecone._client.Pinecone")
def test_cleanup_reads_every_backup_page_before_deleting_any(
    mock_pinecone_cls: MagicMock, projects: Projects
) -> None:
    """The listing is fully walked first, then deleted.

    The page token names a position in a live result set, so a delete
    interleaved with the walk would shift the rows the next token points at.
    """
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    pages = iter([_backup_page("bk-001", next_token="page-2"), _backup_page("bk-002")])
    calls: list[str] = []

    def record_list(**kwargs: object) -> BackupList:
        calls.append("list")
        return next(pages)

    mock_pc.backups.list.side_effect = record_list
    mock_pc.backups.delete.side_effect = lambda **kwargs: calls.append("delete")

    projects._cleanup_project_resources(api_key="test-project-key")

    assert calls == ["list", "list", "delete", "delete"]


@patch("pinecone._client.Pinecone")
def test_cleanup_stops_paging_when_next_token_is_null(
    mock_pinecone_cls: MagicMock, projects: Projects
) -> None:
    """A pagination envelope carrying ``next=None`` ends the walk."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.backups.list.return_value = BackupList(
        [_make_backup("bk-001")],  # type: ignore[misc]
        pagination=Pagination(next=None),
    )

    projects._cleanup_project_resources(api_key="test-project-key")

    assert len(mock_pc.backups.list.call_args_list) == 1
    mock_pc.backups.delete.assert_called_once_with(backup_id="bk-001")


@patch("pinecone._client.Pinecone")
def test_cleanup_stops_when_the_backup_listing_repeats_a_token(
    mock_pinecone_cls: MagicMock, projects: Projects
) -> None:
    """A server that keeps naming the same next page does not hang cleanup.

    ``delete_with_cleanup`` runs unattended, so a walk that never ends is
    worse than one that stops and lets the project delete report what is
    still there.
    """
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.backups.list.side_effect = [
        _backup_page("bk-001", next_token="stuck"),
        _backup_page("bk-002", next_token="stuck"),
    ]

    projects._cleanup_project_resources(api_key="test-project-key")

    assert len(mock_pc.backups.list.call_args_list) == 2
    deleted = [kwargs["backup_id"] for _, kwargs in mock_pc.backups.delete.call_args_list]
    assert deleted == ["bk-001", "bk-002"]


@patch("pinecone._client.Pinecone")
def test_cleanup_deletes_a_repeated_backup_once(
    mock_pinecone_cls: MagicMock, projects: Projects
) -> None:
    """A backup the server reports on two pages is deleted once.

    Paging a live result set can hand the same row back twice; a second
    delete of it would raise nothing but waste a request.
    """
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.backups.list.side_effect = [
        _backup_page("bk-001", "bk-002", next_token="page-2"),
        _backup_page("bk-002", "bk-003"),
    ]

    projects._cleanup_project_resources(api_key="test-project-key")

    deleted = [kwargs["backup_id"] for _, kwargs in mock_pc.backups.delete.call_args_list]
    assert deleted == ["bk-001", "bk-002", "bk-003"]


@patch("pinecone._client.Pinecone")
def test_cleanup_deletes_collections(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify delete is called for each collection returned by list."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.collections.list.return_value = [_make_collection("col-a"), _make_collection("col-b")]

    projects._cleanup_project_resources(api_key="test-project-key")

    assert mock_pc.collections.delete.call_count == 2
    mock_pc.collections.delete.assert_any_call("col-a")
    mock_pc.collections.delete.assert_any_call("col-b")
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_deletes_assistants(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify delete is called for each assistant returned by list."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.assistants.list.return_value = [
        _make_assistant("assistant-a"),
        _make_assistant("assistant-b"),
    ]

    projects._cleanup_project_resources(api_key="test-project-key")

    assert mock_pc.assistants.delete.call_count == 2
    mock_pc.assistants.delete.assert_any_call(name="assistant-a")
    mock_pc.assistants.delete.assert_any_call(name="assistant-b")
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_waits_for_each_assistant_to_disappear(
    mock_pinecone_cls: MagicMock, projects: Projects
) -> None:
    """The delete must not opt out of polling.

    An assistant that has only been asked to terminate still counts against
    the project, so passing ``timeout=-1`` here would reintroduce the 412
    this cleanup exists to avoid.
    """
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.assistants.list.return_value = [_make_assistant("assistant-a")]

    projects._cleanup_project_resources(api_key="test-project-key")

    _, kwargs = mock_pc.assistants.delete.call_args
    assert kwargs.get("timeout") is None


@patch("pinecone._client.Pinecone")
def test_cleanup_follows_server_blocker_order(
    mock_pinecone_cls: MagicMock, projects: Projects
) -> None:
    """Resources are cleared in the order the project delete checks them."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)
    mock_pc.indexes.list.return_value = [_make_index("index-1")]
    mock_pc.collections.list.return_value = [_make_collection("col-a")]
    mock_pc.assistants.list.return_value = [_make_assistant("assistant-a")]
    mock_pc.backups.list.return_value = _backup_page("bk-001")

    order: list[str] = []
    mock_pc.indexes.delete.side_effect = lambda *a, **kw: order.append("index")
    mock_pc.collections.delete.side_effect = lambda *a, **kw: order.append("collection")
    mock_pc.assistants.delete.side_effect = lambda *a, **kw: order.append("assistant")
    mock_pc.backups.delete.side_effect = lambda *a, **kw: order.append("backup")

    projects._cleanup_project_resources(api_key="test-project-key")

    assert order == ["index", "collection", "assistant", "backup"]


@patch("pinecone._client.Pinecone")
def test_cleanup_ignores_not_found(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify NotFoundError during deletion is swallowed, not propagated."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    mock_pc.indexes.list.return_value = [_make_index("index-gone")]
    mock_pc.collections.list.return_value = [_make_collection("col-gone")]
    mock_pc.assistants.list.return_value = [_make_assistant("assistant-gone")]
    mock_pc.backups.list.return_value = _backup_page("bk-gone")

    mock_pc.indexes.delete.side_effect = NotFoundError(message="not found")
    mock_pc.collections.delete.side_effect = NotFoundError(message="not found")
    mock_pc.assistants.delete.side_effect = NotFoundError(message="not found")
    mock_pc.backups.delete.side_effect = NotFoundError(message="not found")

    # Should not raise
    projects._cleanup_project_resources(api_key="test-project-key")

    mock_pc.indexes.delete.assert_called_once_with("index-gone")
    mock_pc.collections.delete.assert_called_once_with("col-gone")
    mock_pc.assistants.delete.assert_called_once_with(name="assistant-gone")
    mock_pc.backups.delete.assert_called_once_with(backup_id="bk-gone")
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_empty_project(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify cleanup completes successfully when project has no resources."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    _empty_project(mock_pc)

    projects._cleanup_project_resources(api_key="test-project-key")

    mock_pc.indexes.delete.assert_not_called()
    mock_pc.collections.delete.assert_not_called()
    mock_pc.assistants.delete.assert_not_called()
    mock_pc.backups.delete.assert_not_called()
    mock_pc.close.assert_called_once()


@patch("pinecone._client.Pinecone")
def test_cleanup_closes_client_on_error(mock_pinecone_cls: MagicMock, projects: Projects) -> None:
    """Verify the Pinecone client is closed even if an unexpected error occurs."""
    mock_pc = MagicMock()
    mock_pinecone_cls.return_value = mock_pc

    mock_pc.indexes.list.side_effect = RuntimeError("unexpected")

    with pytest.raises(RuntimeError, match="unexpected"):
        projects._cleanup_project_resources(api_key="test-project-key")

    mock_pc.close.assert_called_once()
