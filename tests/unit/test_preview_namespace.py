"""Tests for the preview namespace wiring on Pinecone and AsyncPinecone."""

from __future__ import annotations

from pinecone import AsyncPinecone, Pinecone
from pinecone.preview import AsyncPreview, Preview


class TestSyncPreview:
    def test_preview_returns_preview_instance(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert isinstance(pc.preview, Preview)

    def test_preview_is_lazily_constructed(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert pc._preview is None
        first = pc.preview
        second = pc.preview
        assert first is second

    def test_preview_repr(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert repr(pc.preview) == "Preview()"


class TestAsyncPreview:
    def test_preview_returns_async_preview_instance(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        assert isinstance(pc.preview, AsyncPreview)

    def test_preview_is_lazily_constructed(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        assert pc._preview is None
        first = pc.preview
        second = pc.preview
        assert first is second

    def test_preview_repr(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        assert repr(pc.preview) == "AsyncPreview()"
