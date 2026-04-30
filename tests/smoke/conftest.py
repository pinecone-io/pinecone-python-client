"""Shared fixtures and configuration for the smoke-test suite.

Smoke tests are notebook-style end-to-end scripts that walk through one
scenario each, exercising every method in the integration-testing punchlist
at least once on the happy path. They share infrastructure with
``tests/integration/`` (API key, polling helpers, cleanup utilities) but
live in their own directory so they can be run independently.

Run all smoke tests::

    PINECONE_API_KEY=... uv run --with python-dotenv pytest tests/smoke/ -v -s

Run only the fastest priority-1+2 path::

    pytest tests/smoke/test_inference_sync.py tests/smoke/test_inference_async.py \\
           tests/smoke/test_deprecated_shims_sync.py tests/smoke/test_deprecated_shims_async.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from pinecone import Pinecone

# Re-export shared helpers so smoke tests can import them from this conftest.
from tests.integration.conftest import (  # noqa: F401 — re-exported for tests
    async_cleanup_resource,
    async_client,
    async_ensure_index_deleted,
    async_poll_until,
    cleanup_resource,
    ensure_index_deleted,
    poll_until,
    unique_name,
    wait_for_ready,
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "smoke: end-to-end smoke tests that hit a real Pinecone backend",
    )


# Load .env from the SDK root so PINECONE_API_KEY is available even when this
# directory is the only one collected.
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_key() -> str:
    """Pinecone API key from environment. Skips smoke tests if absent."""
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        pytest.skip("PINECONE_API_KEY not set")
    return key


@pytest.fixture
def client(api_key: str) -> Pinecone:
    """Function-scoped sync client.

    Function scope (not session) keeps each smoke scenario isolated — a client
    closed inside one test must not affect the next.
    """
    return Pinecone(api_key=api_key)


# ---------------------------------------------------------------------------
# Smoke prefix — make orphan detection trivial
# ---------------------------------------------------------------------------

SMOKE_PREFIX = "smoke"
"""All resources created by smoke tests must start with this prefix.

The orphan-cleanup script uses this prefix to find and delete any resources
left behind by killed jobs.
"""
