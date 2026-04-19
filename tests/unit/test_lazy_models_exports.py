"""Tests for lazy __getattr__ and __dir__ in pinecone.models sub-packages."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path, known_attr, expected_origin",
    [
        ("pinecone.models.vectors", "Vector", "pinecone.models.vectors.vector"),
        ("pinecone.models.vectors", "SparseValues", "pinecone.models.vectors.sparse"),
        ("pinecone.models.indexes", "IndexModel", "pinecone.models.indexes.index"),
        ("pinecone.models.indexes", "ServerlessSpec", "pinecone.models.indexes.specs"),
        ("pinecone.models", "BatchError", "pinecone.models.batch"),
    ],
)
def test_lazy_models_getattr_returns_class_from_expected_submodule(
    module_path: str, known_attr: str, expected_origin: str | None
) -> None:
    mod = importlib.import_module(module_path)
    obj = getattr(mod, known_attr)
    assert obj is not None
    assert obj.__name__ == known_attr
    if expected_origin is not None:
        assert obj.__module__ == expected_origin


@pytest.mark.parametrize(
    "module_path",
    [
        "pinecone.models.vectors",
        "pinecone.models.indexes",
        "pinecone.models",
    ],
)
def test_lazy_models_getattr_unknown_raises_attribute_error(
    module_path: str,
) -> None:
    mod = importlib.import_module(module_path)
    with pytest.raises(AttributeError, match=r"has no attribute 'NotAReal'"):
        mod.NotAReal  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "module_path",
    [
        "pinecone.models.vectors",
        "pinecone.models.indexes",
        "pinecone.models",
    ],
)
def test_lazy_models_dir_includes_lazy_names(module_path: str) -> None:
    mod = importlib.import_module(module_path)
    assert set(mod._LAZY_IMPORTS).issubset(set(dir(mod)))  # type: ignore[attr-defined]
