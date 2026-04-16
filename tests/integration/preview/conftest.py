"""Shared fixtures for preview integration tests.

These tests make real API calls to Pinecone's preview endpoints
(API version 2026-01.alpha) and require a .env file at the SDK root
with PINECONE_API_KEY set:

    echo 'PINECONE_API_KEY=your-api-key' > .env
    cd sdks/python-sdk2 && uv run --with python-dotenv pytest tests/integration/preview/ -v -s

Unlike the stable integration tests, preview tests skip gracefully
(rather than failing) when the 2026-01.alpha endpoint is unavailable
for the test project's region or account. Preview integration tests do
NOT gate CI.

Each test module that uses preview fixtures must apply:
    pytestmark = pytest.mark.preview_integration

This conftest does NOT apply that mark implicitly — each test module
is responsible for marking itself.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio

from pinecone import AsyncPinecone, Pinecone
from tests.integration.conftest import (
    async_cleanup_resource,
    cleanup_resource,
    unique_name,
)

_preview_available: bool | None = None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "preview_integration: real-API tests against 2026-01.alpha",
    )


@pytest.fixture(scope="session")
def preview_available(client: Pinecone) -> bool:
    """Probe whether the 2026-01.alpha preview endpoint is available.

    Calls client.preview.indexes.list(limit=1) once per session.
    Returns True on success, False on any error (4xx/5xx/connection).
    Result is cached in a module-level variable to avoid repeated probes.
    """
    global _preview_available
    if _preview_available is not None:
        return _preview_available
    try:
        client.preview.indexes.list(limit=1)
        _preview_available = True
    except Exception:
        _preview_available = False
    return _preview_available


@pytest.fixture
def require_preview(preview_available: bool) -> None:
    """Skip the test if the preview endpoint is unavailable."""
    if not preview_available:
        pytest.skip("preview endpoint unavailable (2026-01.alpha not enabled for this project)")


@pytest.fixture
def preview_namespace() -> str:
    """Unique namespace per test for isolation."""
    return f"preview-ns-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def preview_index_name() -> str:
    """Unique index name per test."""
    return unique_name("preview-idx")


@pytest.fixture
def cleanup_preview_indexes(client: Pinecone) -> Generator[list[str], None, None]:
    """Yield a list; on teardown, delete all named preview indexes."""
    names: list[str] = []
    yield names
    for name in names:
        cleanup_resource(
            lambda n=name: client.preview.indexes.delete(n),
            name,
            "preview index",
        )


@pytest_asyncio.fixture
async def async_cleanup_preview_indexes(
    async_client: AsyncPinecone,
) -> AsyncGenerator[list[str], None]:
    """Async mirror of cleanup_preview_indexes."""
    names: list[str] = []
    yield names
    for name in names:
        await async_cleanup_resource(
            lambda n=name: async_client.preview.indexes.delete(n),
            name,
            "preview index",
        )
