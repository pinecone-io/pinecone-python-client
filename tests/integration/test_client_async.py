"""Integration tests for async Pinecone client initialization (REST async)."""

from __future__ import annotations

import pytest
from pinecone import AsyncPinecone


@pytest.mark.integration
async def test_async_client_init_with_api_key(async_client: AsyncPinecone) -> None:
    """AsyncPinecone(api_key=...) creates a client with accessible control-plane namespaces."""
    pc = async_client

    # Client should have the control-plane namespaces accessible
    assert pc.indexes is not None
    assert pc.inference is not None

    # version should be a non-empty string
    from pinecone import __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0
