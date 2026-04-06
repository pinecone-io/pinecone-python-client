"""Unit tests for AsyncPinecone client."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.errors.exceptions import ValidationError


def test_async_pinecone_requires_api_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValidationError, match="No API key"):
            AsyncPinecone()


def test_async_pinecone_accepts_api_key() -> None:
    pc = AsyncPinecone(api_key="test-key")
    assert pc.config.api_key == "test-key"


def test_async_pinecone_deprecated_kwargs() -> None:
    with pytest.raises(ValidationError, match="no longer supported"):
        AsyncPinecone(api_key="test-key", openapi_config={})


def test_async_pinecone_default_host() -> None:
    pc = AsyncPinecone(api_key="test-key")
    assert pc.config.host == "https://api.pinecone.io"


def test_async_pinecone_custom_host() -> None:
    pc = AsyncPinecone(api_key="test-key", host="https://custom.pinecone.io")
    assert pc.config.host == "https://custom.pinecone.io"


def test_async_pinecone_indexes_property() -> None:
    pc = AsyncPinecone(api_key="test-key")
    from pinecone.async_client.indexes import AsyncIndexes

    indexes = pc.indexes
    assert isinstance(indexes, AsyncIndexes)
    # Verify lazy caching — same instance returned
    assert pc.indexes is indexes


async def test_async_pinecone_context_manager() -> None:
    async with AsyncPinecone(api_key="test-key") as pc:
        assert pc.config.api_key == "test-key"


async def test_async_pinecone_close() -> None:
    pc = AsyncPinecone(api_key="test-key")
    await pc.close()
