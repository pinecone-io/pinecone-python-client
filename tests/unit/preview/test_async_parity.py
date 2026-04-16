"""Async parity tests: verify AsyncPreviewDocuments and AsyncPreviewIndexes mirror their
sync counterparts exactly — same parameter names, kinds, defaults, and annotations —
and that validation logic, error messages, and batch_upsert return shapes are identical.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Coroutine
from typing import Any, get_args, get_origin
from unittest.mock import AsyncMock, MagicMock

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.errors.exceptions import ValidationError
from pinecone.models.batch import BatchResult
from pinecone.models.pagination import AsyncPaginator, Paginator
from pinecone.preview.async_documents import AsyncPreviewDocuments
from pinecone.preview.async_index import AsyncPreviewIndex
from pinecone.preview.async_indexes import AsyncPreviewIndexes
from pinecone.preview.documents import PreviewDocuments
from pinecone.preview.index import PreviewIndex
from pinecone.preview.indexes import PreviewIndexes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_coroutine(annotation: Any) -> Any:
    """Return T if annotation is Coroutine[Any, Any, T], else the annotation unchanged."""
    if get_origin(annotation) is Coroutine:
        args = get_args(annotation)
        if args:
            return args[2]
    return annotation


def _normalize_paginator(annotation: Any) -> Any:
    """Convert AsyncPaginator[T] → Paginator[T] for comparison purposes.

    Handles both string annotations (from ``from __future__ import annotations``)
    and resolved generic aliases.
    """
    if isinstance(annotation, str) and annotation.startswith("AsyncPaginator["):
        return "Paginator[" + annotation[len("AsyncPaginator["):]
    if get_origin(annotation) is AsyncPaginator:
        inner = get_args(annotation)
        return Paginator[inner[0]] if inner else Paginator  # type: ignore[valid-type]
    return annotation


def _param_pairs(
    sync_cls: type, async_cls: type, method_name: str
) -> tuple[dict[str, inspect.Parameter], dict[str, inspect.Parameter]]:
    sync_sig = inspect.signature(getattr(sync_cls, method_name))
    async_sig = inspect.signature(getattr(async_cls, method_name))
    return dict(sync_sig.parameters), dict(async_sig.parameters)


def _assert_signature_parity(sync_cls: type, async_cls: type, method_name: str) -> None:
    sync_params, async_params = _param_pairs(sync_cls, async_cls, method_name)

    assert set(sync_params) == set(async_params), (
        f"{method_name}: parameter names differ — "
        f"sync-only={set(sync_params) - set(async_params)}, "
        f"async-only={set(async_params) - set(sync_params)}"
    )

    for name in sync_params:
        sp = sync_params[name]
        ap = async_params[name]

        assert sp.kind == ap.kind, (
            f"{method_name}.{name}: kind differs (sync={sp.kind}, async={ap.kind})"
        )
        assert sp.default == ap.default, (
            f"{method_name}.{name}: default differs "
            f"(sync={sp.default!r}, async={ap.default!r})"
        )

        s_ann = sp.annotation if sp.annotation is not inspect.Parameter.empty else None
        a_ann = ap.annotation if ap.annotation is not inspect.Parameter.empty else None
        if s_ann is not None and a_ann is not None:
            assert s_ann == a_ann, (
                f"{method_name}.{name}: annotation differs (sync={s_ann}, async={a_ann})"
            )


def _assert_return_parity(sync_cls: type, async_cls: type, method_name: str) -> None:
    sync_sig = inspect.signature(getattr(sync_cls, method_name))
    async_sig = inspect.signature(getattr(async_cls, method_name))

    s_ret = sync_sig.return_annotation
    a_ret = async_sig.return_annotation
    if s_ret is inspect.Parameter.empty or a_ret is inspect.Parameter.empty:
        return

    a_ret_unwrapped = _normalize_paginator(_strip_coroutine(a_ret))
    s_ret_normalized = _normalize_paginator(s_ret)
    assert s_ret_normalized == a_ret_unwrapped, (
        f"{method_name}: return annotation differs "
        f"(sync={s_ret}, async unwrapped={a_ret_unwrapped})"
    )


# ---------------------------------------------------------------------------
# PreviewDocuments vs AsyncPreviewDocuments — signature parity
# ---------------------------------------------------------------------------

_DOCUMENTS_METHODS = ["upsert", "batch_upsert", "search", "fetch", "delete"]


@pytest.mark.parametrize("method_name", _DOCUMENTS_METHODS)
def test_documents_parameter_parity(method_name: str) -> None:
    _assert_signature_parity(PreviewDocuments, AsyncPreviewDocuments, method_name)


@pytest.mark.parametrize("method_name", _DOCUMENTS_METHODS)
def test_documents_return_annotation_parity(method_name: str) -> None:
    _assert_return_parity(PreviewDocuments, AsyncPreviewDocuments, method_name)


# ---------------------------------------------------------------------------
# PreviewIndexes vs AsyncPreviewIndexes — signature parity
# ---------------------------------------------------------------------------

_INDEXES_METHODS = [
    "create",
    "configure",
    "describe",
    "list",
    "exists",
    "delete",
    "create_backup",
    "list_backups",
]


@pytest.mark.parametrize("method_name", _INDEXES_METHODS)
def test_indexes_parameter_parity(method_name: str) -> None:
    _assert_signature_parity(PreviewIndexes, AsyncPreviewIndexes, method_name)


@pytest.mark.parametrize("method_name", _INDEXES_METHODS)
def test_indexes_return_annotation_parity(method_name: str) -> None:
    _assert_return_parity(PreviewIndexes, AsyncPreviewIndexes, method_name)


# ---------------------------------------------------------------------------
# Validation parity — error messages must match between sync and async
# ---------------------------------------------------------------------------


def _make_config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key", host="https://api.test.pinecone.io")


def _make_sync_docs() -> PreviewDocuments:
    config = _make_config()
    http = HTTPClient(config, "2026-01.alpha")
    return PreviewDocuments(http=http, config=config, host="https://host.test")


def _make_async_docs() -> AsyncPreviewDocuments:
    config = _make_config()
    http = AsyncHTTPClient(config, "2026-01.alpha")
    return AsyncPreviewDocuments(http=http, config=config, host="https://host.test")


def _sync_error(fn: Any, *args: Any, **kwargs: Any) -> str:
    with pytest.raises(ValidationError) as exc_info:
        fn(*args, **kwargs)
    return str(exc_info.value)


async def _async_error(fn: Any, *args: Any, **kwargs: Any) -> str:
    with pytest.raises(ValidationError) as exc_info:
        await fn(*args, **kwargs)
    return str(exc_info.value)


# fetch validation

def test_fetch_empty_namespace_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.fetch, namespace="")
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.fetch, namespace="")
    )
    assert sync_msg == async_msg


# search validation

def test_search_empty_namespace_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.search, namespace="", top_k=10, score_by=[{"bm25": {}}])
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.search, namespace="", top_k=10, score_by=[{"bm25": {}}])
    )
    assert sync_msg == async_msg


def test_search_top_k_range_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.search, namespace="ns", top_k=0, score_by=[{"bm25": {}}])
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.search, namespace="ns", top_k=0, score_by=[{"bm25": {}}])
    )
    assert sync_msg == async_msg


def test_search_empty_score_by_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.search, namespace="ns", top_k=5, score_by=[])
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.search, namespace="ns", top_k=5, score_by=[])
    )
    assert sync_msg == async_msg


# delete validation

def test_delete_empty_namespace_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.delete, namespace="", ids=["a"])
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.delete, namespace="", ids=["a"])
    )
    assert sync_msg == async_msg


def test_delete_no_target_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.delete, namespace="ns")
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.delete, namespace="ns")
    )
    assert sync_msg == async_msg


def test_delete_ids_and_delete_all_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(sync.delete, namespace="ns", ids=["a"], delete_all=True)
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.delete, namespace="ns", ids=["a"], delete_all=True)
    )
    assert sync_msg == async_msg


def test_delete_ids_and_filter_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    sync_msg = _sync_error(
        sync.delete, namespace="ns", ids=["a"], filter={"field": {"$eq": "v"}}
    )
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(
            async_d.delete, namespace="ns", ids=["a"], filter={"field": {"$eq": "v"}}
        )
    )
    assert sync_msg == async_msg


# upsert validation

def test_upsert_empty_namespace_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"_id": "doc1", "text": "hello"}]
    sync_msg = _sync_error(sync.upsert, namespace="", documents=docs)
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.upsert, namespace="", documents=docs)
    )
    assert sync_msg == async_msg


def test_upsert_missing_id_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"text": "hello"}]
    sync_msg = _sync_error(sync.upsert, namespace="ns", documents=docs)
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.upsert, namespace="ns", documents=docs)
    )
    assert sync_msg == async_msg


def test_upsert_duplicate_id_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"_id": "dup", "text": "a"}, {"_id": "dup", "text": "b"}]
    sync_msg = _sync_error(sync.upsert, namespace="ns", documents=docs)
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.upsert, namespace="ns", documents=docs)
    )
    assert sync_msg == async_msg


# batch_upsert validation

def test_batch_upsert_batch_size_range_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"_id": "d1"}]
    sync_msg = _sync_error(sync.batch_upsert, namespace="ns", documents=docs, batch_size=0)
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.batch_upsert, namespace="ns", documents=docs, batch_size=0)
    )
    assert sync_msg == async_msg


def test_batch_upsert_max_workers_range_parity() -> None:
    sync = _make_sync_docs()
    async_d = _make_async_docs()
    docs = [{"_id": "d1"}]
    sync_msg = _sync_error(sync.batch_upsert, namespace="ns", documents=docs, max_workers=0)
    async_msg = asyncio.get_event_loop().run_until_complete(
        _async_error(async_d.batch_upsert, namespace="ns", documents=docs, max_workers=0)
    )
    assert sync_msg == async_msg


# ---------------------------------------------------------------------------
# batch_upsert returns BatchResult with the same populated fields
# ---------------------------------------------------------------------------

_DOCS_FIXTURE = [{"_id": f"doc{i}", "text": f"text {i}"} for i in range(3)]

_UPSERT_RESPONSE = b'{"upserted_count": 3}'


def test_sync_batch_upsert_returns_batch_result() -> None:
    config = _make_config()
    mock_response = MagicMock()
    mock_response.content = _UPSERT_RESPONSE

    docs_obj = _make_sync_docs()
    docs_obj._http = MagicMock()
    docs_obj._http.post.return_value = mock_response

    result = docs_obj.batch_upsert(
        namespace="ns",
        documents=_DOCS_FIXTURE,
        batch_size=10,
        max_workers=1,
        show_progress=False,
    )

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 3
    assert result.successful_item_count == 3
    assert result.failed_item_count == 0
    assert result.total_batch_count == 1
    assert result.successful_batch_count == 1
    assert result.failed_batch_count == 0
    assert result.errors == []


def test_async_batch_upsert_returns_batch_result() -> None:
    mock_response = MagicMock()
    mock_response.content = _UPSERT_RESPONSE

    async def run() -> BatchResult:
        docs_obj = _make_async_docs()
        docs_obj._http = AsyncMock()
        docs_obj._http.post.return_value = mock_response

        return await docs_obj.batch_upsert(
            namespace="ns",
            documents=_DOCS_FIXTURE,
            batch_size=10,
            max_workers=1,
            show_progress=False,
        )

    result = asyncio.get_event_loop().run_until_complete(run())

    assert isinstance(result, BatchResult)
    assert result.total_item_count == 3
    assert result.successful_item_count == 3
    assert result.failed_item_count == 0
    assert result.total_batch_count == 1
    assert result.successful_batch_count == 1
    assert result.failed_batch_count == 0
    assert result.errors == []


def test_batch_upsert_same_fields_both_variants() -> None:
    """Both sync and async batch_upsert fill the same BatchResult fields."""
    mock_response = MagicMock()
    mock_response.content = _UPSERT_RESPONSE

    sync_docs = _make_sync_docs()
    sync_docs._http = MagicMock()
    sync_docs._http.post.return_value = mock_response
    sync_result = sync_docs.batch_upsert(
        namespace="ns",
        documents=_DOCS_FIXTURE,
        batch_size=10,
        max_workers=1,
        show_progress=False,
    )

    async def run_async() -> BatchResult:
        async_docs = _make_async_docs()
        async_docs._http = AsyncMock()
        async_docs._http.post.return_value = mock_response
        return await async_docs.batch_upsert(
            namespace="ns",
            documents=_DOCS_FIXTURE,
            batch_size=10,
            max_workers=1,
            show_progress=False,
        )

    async_result = asyncio.get_event_loop().run_until_complete(run_async())

    for field in [
        "total_item_count",
        "successful_item_count",
        "failed_item_count",
        "total_batch_count",
        "successful_batch_count",
        "failed_batch_count",
    ]:
        assert getattr(sync_result, field) == getattr(async_result, field), (
            f"BatchResult.{field} differs between sync and async"
        )


# ---------------------------------------------------------------------------
# PreviewIndex.documents and AsyncPreviewIndex.documents are correct types
# ---------------------------------------------------------------------------


def test_preview_index_documents_type() -> None:
    config = _make_config()
    http = HTTPClient(config, "2026-01.alpha")
    idx = PreviewIndex(host="https://host.test", http=http, config=config)
    assert isinstance(idx.documents, PreviewDocuments)


def test_async_preview_index_documents_type() -> None:
    config = _make_config()
    http = AsyncHTTPClient(config, "2026-01.alpha")
    idx = AsyncPreviewIndex(http=http, config=config, host="https://host.test")
    assert isinstance(idx.documents, AsyncPreviewDocuments)


def test_async_preview_index_documents_type_unresolved() -> None:
    """Accessing .documents before host resolution raises RuntimeError."""
    config = _make_config()
    idx = AsyncPreviewIndex(http=MagicMock(), config=config)
    with pytest.raises(RuntimeError):
        _ = idx.documents
