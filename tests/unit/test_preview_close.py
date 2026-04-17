"""Tests for close() propagation through the preview namespace."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pinecone import AsyncPinecone, Pinecone


class TestPreviewIndexesClose:
    def test_close_calls_http_close(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        indexes = pc.preview.indexes
        with patch.object(indexes._http, "close") as mock_close:
            indexes.close()
        mock_close.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        indexes = pc.preview.indexes
        with patch.object(indexes._http, "close") as mock_close:
            indexes.close()
            indexes.close()
        assert mock_close.call_count == 2


class TestAsyncPreviewIndexesClose:
    @pytest.mark.asyncio
    async def test_close_calls_http_close(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        indexes = pc.preview.indexes
        with patch.object(indexes._http, "close", new_callable=AsyncMock) as mock_close:
            await indexes.close()
        mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        indexes = pc.preview.indexes
        with patch.object(indexes._http, "close", new_callable=AsyncMock) as mock_close:
            await indexes.close()
            await indexes.close()
        assert mock_close.call_count == 2


class TestPreviewClose:
    def test_close_with_indexes_initialized(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        _ = pc.preview.indexes  # trigger lazy init
        assert pc.preview._indexes is not None
        with patch.object(pc.preview._indexes, "close") as mock_close:
            pc.preview.close()
        mock_close.assert_called_once()

    def test_close_without_indexes_initialized(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        assert pc.preview._indexes is None
        # should not raise
        pc.preview.close()

    def test_close_is_idempotent(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        _ = pc.preview.indexes
        with patch.object(pc.preview._indexes, "close") as mock_close:  # type: ignore[union-attr]
            pc.preview.close()
            pc.preview.close()
        assert mock_close.call_count == 2


class TestAsyncPreviewClose:
    @pytest.mark.asyncio
    async def test_close_with_indexes_initialized(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        _ = pc.preview.indexes
        assert pc.preview._indexes is not None
        with patch.object(pc.preview._indexes, "close", new_callable=AsyncMock) as mock_close:
            await pc.preview.close()
        mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_indexes_initialized(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        assert pc.preview._indexes is None
        # should not raise
        await pc.preview.close()


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
