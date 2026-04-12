"""Shared test fixtures for the Pinecone SDK test suite."""

from __future__ import annotations

import pytest

from pinecone._internal.config import PineconeConfig


@pytest.fixture
def api_key() -> str:
    """Return a test API key string."""
    return "test-api-key-00000000"


@pytest.fixture
def base_url() -> str:
    """Return a test base URL."""
    return "https://api.test.pinecone.io"


@pytest.fixture
def config(api_key: str, base_url: str) -> PineconeConfig:
    """Return a PineconeConfig with test defaults."""
    return PineconeConfig(
        api_key=api_key,
        host=base_url,
    )
