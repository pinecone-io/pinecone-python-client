"""Tests for close() on PreviewIndexes, AsyncPreviewIndexes, Preview, AsyncPreview."""

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
        with patch.object(pc.preview._indexes, "close") as mock_close:
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
