"""Tests for close() on PreviewDocuments and AsyncPreviewDocuments."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from pinecone import AsyncPinecone, Pinecone
from pinecone._internal.config import PineconeConfig
from pinecone.preview.async_index import AsyncPreviewIndex


class TestPreviewDocumentsClose:
    def test_close_calls_http_close(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        docs = pc.preview.index(host="https://h.example.svc.pinecone.io").documents
        with patch.object(docs._http, "close") as mock_close:
            docs.close()
        mock_close.assert_called_once()

    def test_close_is_idempotent_raises_no_error(self) -> None:
        pc = Pinecone(api_key="test-key-1234")
        docs = pc.preview.index(host="https://h.example.svc.pinecone.io").documents
        with patch.object(docs._http, "close") as mock_close:
            docs.close()
            docs.close()
        assert mock_close.call_count == 2


class TestAsyncPreviewDocumentsClose:
    async def test_close_calls_http_close_when_http_initialized(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        docs = pc.preview.index(host="https://h.example.svc.pinecone.io").documents
        with patch.object(docs._http, "close", new_callable=AsyncMock) as mock_close:
            await docs.close()
        mock_close.assert_called_once()

    async def test_close_is_noop_when_http_not_initialized(self) -> None:
        async def _provider() -> str:
            return "https://lazy-host.svc.pinecone.io"

        idx = AsyncPreviewIndex(
            config=PineconeConfig(api_key="test-key-1234", host="https://api.test.pinecone.io"),
            _host_provider=_provider,
        )
        docs = idx.documents
        assert docs._http is None
        await docs.close()
        assert docs._http is None

    async def test_close_is_idempotent_raises_no_error(self) -> None:
        pc = AsyncPinecone(api_key="test-key-1234")
        docs = pc.preview.index(host="https://h.example.svc.pinecone.io").documents
        with patch.object(docs._http, "close", new_callable=AsyncMock) as mock_close:
            await docs.close()
            await docs.close()
        assert mock_close.call_count == 2
