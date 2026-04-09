"""Shared fixtures for integration tests.

These tests make real API calls to Pinecone and require a .env file
at the SDK root with PINECONE_API_KEY set:

    echo 'PINECONE_API_KEY=your-api-key' > .env
    cd sdks/python-sdk2 && uv run --with python-dotenv pytest tests/integration/ -v -s
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pinecone import Pinecone


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: marks tests as real-API integration tests")

# Load .env from the SDK root (two levels up from tests/integration/)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_ready(
    check_fn: object,
    *,
    timeout: int = 300,
    interval: int = 5,
    description: str = "resource",
) -> None:
    """Poll until check_fn() returns True or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if check_fn():  # type: ignore[operator]
                return
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"{description} not ready after {timeout}s")


def poll_until(
    query_fn: object,
    check_fn: object,
    *,
    timeout: int = 60,
    interval: int = 3,
    description: str = "condition",
) -> object:
    """Poll query_fn() until check_fn(result) is True. Returns the final result."""
    start = time.time()
    last_result = None
    while time.time() - start < timeout:
        try:
            last_result = query_fn()  # type: ignore[operator]
            if check_fn(last_result):  # type: ignore[operator]
                return last_result
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"{description} not satisfied after {timeout}s (last result: {last_result})")


def unique_name(prefix: str = "inttest") -> str:
    """Generate a unique resource name using timestamp + random suffix."""
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}-{int(time.time())}-{short_uuid}"


def cleanup_resource(
    delete_fn: object,
    resource_id: str,
    resource_type: str = "resource",
) -> None:
    """Best-effort cleanup of a named resource. Logs but never raises."""
    try:
        delete_fn()  # type: ignore[operator]
        print(f"  Cleaned up {resource_type}: {resource_id}")
    except Exception as exc:
        print(f"  WARNING: Failed to clean up {resource_type} {resource_id}: {exc}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def api_key() -> str:
    """Pinecone API key from environment. Skips all tests if not set."""
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        pytest.skip("PINECONE_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def client(api_key: str) -> Pinecone:
    """Session-scoped Pinecone client."""
    return Pinecone(api_key=api_key)
