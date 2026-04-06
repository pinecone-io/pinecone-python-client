"""Tests for top-level pinecone package exports."""

from __future__ import annotations


def test_import_integrated_spec() -> None:
    from pinecone import IntegratedSpec

    assert IntegratedSpec is not None


def test_import_embed_config() -> None:
    from pinecone import EmbedConfig

    assert EmbedConfig is not None


def test_import_byoc_spec() -> None:
    from pinecone import ByocSpec

    assert ByocSpec is not None


def test_import_embed_model() -> None:
    from pinecone import EmbedModel

    assert EmbedModel is not None


def test_all_contains_new_exports() -> None:
    import pinecone

    for name in ("IntegratedSpec", "EmbedConfig", "ByocSpec", "EmbedModel"):
        assert name in pinecone.__all__, f"{name} missing from __all__"


def test_dir_contains_new_exports() -> None:
    import pinecone

    module_dir = dir(pinecone)
    for name in ("IntegratedSpec", "EmbedConfig", "ByocSpec", "EmbedModel"):
        assert name in module_dir, f"{name} missing from dir(pinecone)"
