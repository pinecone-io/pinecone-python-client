from __future__ import annotations

import sys
from multiprocessing.pool import ApplyResult  # for isinstance checks
from typing import Any

import pytest

from pinecone._legacy.async_req import install_async_req_support
from pinecone.errors.exceptions import PineconeValueError


class _FakeIndex:
    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.describe_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> str:
        self.upsert_calls.append(kwargs)
        return f"upsert-result:{kwargs}"

    def query(self, **kwargs: Any) -> str:
        self.query_calls.append(kwargs)
        return f"query-result:{kwargs}"

    def describe_index_stats(self, **kwargs: Any) -> str:
        self.describe_calls.append(kwargs)
        return f"describe-result:{kwargs}"

    def list_paginated(self, **kwargs: Any) -> str:
        self.list_calls.append(kwargs)
        return f"list-result:{kwargs}"


def test_install_does_not_import_multiprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Drop any pre-existing import to make the assertion meaningful.
    for mod in list(sys.modules):
        if mod == "multiprocessing.pool" or mod.startswith("multiprocessing.pool."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    idx = _FakeIndex()
    install_async_req_support(idx, pool_threads=2)
    assert "multiprocessing.pool" not in sys.modules


def test_async_req_false_calls_canonical_directly() -> None:
    idx = _FakeIndex()
    install_async_req_support(idx, pool_threads=2)
    result = idx.upsert(vectors=[("a", [0.1])])
    assert result == "upsert-result:{'vectors': [('a', [0.1])]}"
    assert idx.upsert_calls == [{"vectors": [("a", [0.1])]}]


def test_async_req_true_returns_apply_result() -> None:
    idx = _FakeIndex()
    install_async_req_support(idx, pool_threads=2)
    result = idx.upsert(vectors=[("a", [0.1])], async_req=True)
    assert isinstance(result, ApplyResult)
    assert result.get(timeout=5).startswith("upsert-result:")
    # async_req kwarg is stripped before reaching canonical
    assert idx.upsert_calls == [{"vectors": [("a", [0.1])]}]


def test_async_req_true_on_query() -> None:
    idx = _FakeIndex()
    install_async_req_support(idx, pool_threads=2)
    result = idx.query(top_k=5, vector=[0.1], async_req=True)
    assert isinstance(result, ApplyResult)
    assert result.get(timeout=5).startswith("query-result:")


def test_async_req_true_on_describe_index_stats() -> None:
    idx = _FakeIndex()
    install_async_req_support(idx, pool_threads=2)
    result = idx.describe_index_stats(async_req=True)
    assert isinstance(result, ApplyResult)
    assert result.get(timeout=5).startswith("describe-result:")


def test_async_req_true_on_list_paginated() -> None:
    idx = _FakeIndex()
    install_async_req_support(idx, pool_threads=2)
    result = idx.list_paginated(prefix="x#", async_req=True)
    assert isinstance(result, ApplyResult)
    assert result.get(timeout=5).startswith("list-result:")


def test_upsert_batch_size_with_async_req_raises_legacy_text() -> None:
    idx = _FakeIndex()
    install_async_req_support(idx, pool_threads=2)
    with pytest.raises(PineconeValueError) as excinfo:
        idx.upsert(vectors=[("a", [0.1])], batch_size=10, async_req=True)
    assert str(excinfo.value) == ("async_req is not supported when batch_size is provided.")


def test_pool_only_constructed_on_first_async_req_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mod in list(sys.modules):
        if mod == "multiprocessing.pool" or mod.startswith("multiprocessing.pool."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    idx = _FakeIndex()
    install_async_req_support(idx, pool_threads=2)
    # Several non-async calls should not import multiprocessing.pool.
    idx.upsert(vectors=[("a", [0.1])])
    idx.query(top_k=1, vector=[0.1])
    idx.describe_index_stats()
    idx.list_paginated()
    assert "multiprocessing.pool" not in sys.modules
    # First async call triggers the import.
    result = idx.upsert(vectors=[("b", [0.2])], async_req=True)
    result.get(timeout=5)
    assert "multiprocessing.pool" in sys.modules


def test_invalid_pool_threads_rejected() -> None:
    idx = _FakeIndex()
    with pytest.raises(PineconeValueError):
        install_async_req_support(idx, pool_threads=0)
    with pytest.raises(PineconeValueError):
        install_async_req_support(idx, pool_threads=-1)
