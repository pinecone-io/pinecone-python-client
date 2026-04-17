"""Tests for close() propagation through the Pinecone client to the preview namespace."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pinecone import AsyncPinecone, Pinecone


class TestPineconeClosePropagatesToPreview:
    def test_close_propagates_when_preview_accessed(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        preview = pc.preview  # trigger lazy init
        with patch.object(preview, "close") as mock_close:
            pc.close()
        mock_close.assert_called_once()

    def test_close_skips_preview_when_not_accessed(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        assert pc._preview is None
        # should not raise
        pc.close()


class TestAsyncPineconeClosePropagatesToPreview:
    @pytest.mark.asyncio
    async def test_close_propagates_when_preview_accessed(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        preview = pc.preview
        with patch.object(preview, "close", new_callable=AsyncMock) as mock_close:
            await pc.close()
        mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_skips_preview_when_not_accessed(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        assert pc._preview is None
        # should not raise
        await pc.close()


class TestPreviewIndexClose:
    def test_preview_index_close_closes_documents_http(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        idx = pc.preview.index(host="https://test.svc.pinecone.io")
        with patch.object(idx.documents, "close") as mock_close:
            idx.close()
        mock_close.assert_called_once()

    def test_preview_index_context_manager(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        idx = pc.preview.index(host="https://test.svc.pinecone.io")
        with patch.object(idx.documents, "close") as mock_close, idx:
            pass
        mock_close.assert_called_once()

    def test_preview_index_double_close_is_safe(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        idx = pc.preview.index(host="https://test.svc.pinecone.io")
        with patch.object(idx.documents, "close"):
            idx.close()
            idx.close()  # must not raise


class TestAsyncPreviewIndexClose:
    @pytest.mark.asyncio
    async def test_async_preview_index_close_closes_documents_http(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        idx = pc.preview.index(host="https://test.svc.pinecone.io")
        with patch.object(idx._documents, "close", new_callable=AsyncMock) as mock_close:
            await idx.close()
        mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_preview_index_context_manager(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        idx = pc.preview.index(host="https://test.svc.pinecone.io")
        with patch.object(idx._documents, "close", new_callable=AsyncMock) as mock_close:
            async with idx:
                pass
        mock_close.assert_called_once()
