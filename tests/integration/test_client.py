"""Integration tests for Pinecone client initialization."""

from __future__ import annotations

import os

import pytest

from pinecone import Pinecone


@pytest.mark.integration
def test_client_init_with_api_key() -> None:
    """Pinecone(api_key=...) creates a client with accessible control-plane namespaces."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        pytest.skip("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)

    # Client should have the control-plane namespaces accessible
    assert pc.indexes is not None
    assert pc.inference is not None

    # version should be a non-empty string
    from pinecone import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
