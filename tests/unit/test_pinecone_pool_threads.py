"""Unit tests for Pinecone(pool_threads=N) propagation to Index (BC-0110)."""

from __future__ import annotations

import sys
import warnings

import pytest

from pinecone import Pinecone


def test_pinecone_without_pool_threads_does_not_import_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "pinecone._legacy.async_req", raising=False)
    Pinecone(api_key="k")
    assert "pinecone._legacy.async_req" not in sys.modules


def test_pinecone_pool_threads_propagates_to_attribute() -> None:
    pc = Pinecone(api_key="k")
    assert pc._legacy_pool_threads is None  # type: ignore[attr-defined]

    pc2 = Pinecone(api_key="k", pool_threads=8)
    assert pc2._legacy_pool_threads == 8  # type: ignore[attr-defined]


def test_pinecone_index_factory_passes_pool_threads() -> None:
    pc = Pinecone(api_key="k", pool_threads=4)
    kwargs = pc._build_index_kwargs("test.svc.pinecone.io")  # type: ignore[attr-defined]
    assert kwargs["pool_threads"] == 4


def test_pinecone_without_pool_threads_omits_field() -> None:
    pc = Pinecone(api_key="k")
    kwargs = pc._build_index_kwargs("test.svc.pinecone.io")  # type: ignore[attr-defined]
    assert "pool_threads" not in kwargs


def test_pinecone_pool_threads_emits_no_warning() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Pinecone(api_key="k", pool_threads=4)
    assert [w for w in record if issubclass(w.category, DeprecationWarning)] == []


def test_pinecone_index_with_pool_threads_installs_async_req() -> None:
    pc = Pinecone(api_key="k")
    idx = pc.index(host="test.svc.pinecone.io")
    assert not hasattr(idx, "_legacy_async_pool")

    pc2 = Pinecone(api_key="k", pool_threads=4)
    idx2 = pc2.index(host="test.svc.pinecone.io")
    assert hasattr(idx2, "_legacy_async_pool")
    assert idx2._legacy_async_pool_threads == 4  # type: ignore[attr-defined]
