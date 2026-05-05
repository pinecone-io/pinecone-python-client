"""Integration tests for Pinecone client initialization."""

from __future__ import annotations

import os

import httpx
import pytest
import respx

from pinecone import Pinecone
from pinecone.models.backups.model import CreateIndexFromBackupResponse


@pytest.mark.integration
def test_validation_error_import_triggers_deprecation_warning() -> None:
    """Accessing pinecone.ValidationError triggers DeprecationWarning and returns PineconeValueError.

    Verifies unified-depr-0004 backward-compatibility contract: the ValidationError
    alias from v6 SDK remains accessible via the top-level module, BUT raises a
    DeprecationWarning directing callers to PineconeValueError.

    The shim in pinecone/__init__.__getattr__ fires warnings.warn(...,
    DeprecationWarning) on first access and then caches the result.  This test
    clears the cached entry before the assertion so the warning fires fresh even
    if a prior test or import already accessed pinecone.ValidationError.

    Assertions:
    - DeprecationWarning is raised with the expected message
    - The returned class is identical to PineconeValueError (same object)
    - ValidationError is a subclass of ValueError (for backward-compat catch blocks)
    - A PineconeValueError instance can be caught using the deprecated alias
    """
    import pinecone as pinecone_module
    from pinecone.errors.exceptions import PineconeValueError

    # Clear the cached entry so __getattr__ fires again and re-emits the warning.
    # (After the first access globals()["ValidationError"] is set, bypassing __getattr__.)
    pinecone_module.__dict__.pop("ValidationError", None)

    with pytest.warns(DeprecationWarning, match="ValidationError is deprecated"):
        deprecated_cls = pinecone_module.ValidationError

    # The alias must resolve to the same class object as PineconeValueError
    assert deprecated_cls is PineconeValueError, (
        "pinecone.ValidationError must be the same object as PineconeValueError"
    )

    # Must remain a proper ValueError subclass for old except-ValidationError blocks
    assert issubclass(deprecated_cls, ValueError), (
        "ValidationError must be a subclass of ValueError for backward-compatibility"
    )

    # Old code that catches ValidationError must catch PineconeValueError instances
    try:
        raise PineconeValueError("backward-compatibility catch test")
    except deprecated_cls:
        pass  # Expected — the alias catches PineconeValueError
    else:
        pytest.fail("ValidationError alias failed to catch a PineconeValueError instance")


BASE_URL = "https://api.pinecone.io"


@respx.mock
def test_create_index_from_backup_no_wait_returns_restore_job_id() -> None:
    """timeout=-1 returns CreateIndexFromBackupResponse with restore_job_id and index_id."""
    respx.post(f"{BASE_URL}/backups/bk-test/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-test-123", "index_id": "idx-test-456"},
        ),
    )

    pc = Pinecone(api_key="test-key")
    result = pc.create_index_from_backup(
        name="test-restore-nowait",
        backup_id="bk-test",
        timeout=-1,
    )

    assert isinstance(result, CreateIndexFromBackupResponse)
    assert result.restore_job_id == "rj-test-123"
    assert result.index_id == "idx-test-456"


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
