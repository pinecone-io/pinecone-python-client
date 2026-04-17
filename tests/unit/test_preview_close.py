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
