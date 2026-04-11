"""Tests that __dir__ includes lazily-loaded names before first access."""

from __future__ import annotations

import importlib
import sys


def _fresh_import(module_name: str) -> object:
    """Import a module after removing it (and sub-modules) from sys.modules."""
    to_remove = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
    saved = {k: sys.modules.pop(k) for k in to_remove}
    try:
        return importlib.import_module(module_name)
    finally:
        # Restore so other tests aren't affected
        sys.modules.update(saved)


def test_pinecone_dir_includes_lazy_names() -> None:
    """dir(pinecone) should list lazy-loaded classes even before they are accessed."""
    import pinecone

    names = dir(pinecone)
    lazy_names = {
        "Pinecone",
        "AsyncPinecone",
        "Index",
        "AsyncIndex",
        "IndexModel",
        "IndexList",
        "CollectionModel",
        "CollectionList",
        "ServerlessSpec",
        "PodSpec",
        "QueryResponse",
    }
    for name in lazy_names:
        assert name in names, f"{name!r} missing from dir(pinecone)"


def test_pinecone_dir_includes_eagerly_loaded() -> None:
    """dir(pinecone) should also include eagerly-loaded names."""
    import pinecone

    names = dir(pinecone)
    eager_names = {
        "PineconeConfig",
        "ApiError",
        "PineconeError",
        "Metric",
        "CloudProvider",
        "__version__",
    }
    for name in eager_names:
        assert name in names, f"{name!r} missing from dir(pinecone)"


def test_indexes_subpackage_dir() -> None:
    from pinecone.models import indexes

    names = dir(indexes)
    expected = {"IndexModel", "IndexStatus", "IndexList", "ServerlessSpec", "PodSpec", "ByocSpec"}
    for name in expected:
        assert name in names, f"{name!r} missing from dir(pinecone.models.indexes)"


def test_vectors_subpackage_dir() -> None:
    from pinecone.models import vectors

    names = dir(vectors)
    expected = {
        "SparseValues",
        "Usage",
        "Vector",
        "ScoredVector",
        "UpsertResponse",
        "QueryResponse",
        "FetchResponse",
        "NamespaceSummary",
        "DescribeIndexStatsResponse",
        "ListItem",
        "ListResponse",
        "Pagination",
        "UpdateResponse",
    }
    for name in expected:
        assert name in names, f"{name!r} missing from dir(pinecone.models.vectors)"


def test_collections_subpackage_dir() -> None:
    from pinecone.models import collections

    names = dir(collections)
    expected = {"CollectionModel", "CollectionList"}
    for name in expected:
        assert name in names, f"{name!r} missing from dir(pinecone.models.collections)"
