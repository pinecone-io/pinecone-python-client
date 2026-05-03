"""Integration tests for Admin client credential validation.

These tests verify that Admin.__init__ raises typed exceptions for missing,
empty, or whitespace-only credentials, and raises ApiError when the OAuth
endpoint rejects invalid credentials.

The first four tests (init_validation_*) do NOT require real service-account
credentials and run regardless of environment.  They use monkeypatch to clear
any PINECONE_CLIENT_ID / PINECONE_CLIENT_SECRET env vars and pass bogus or
whitespace-only values so execution exits before any network call.

The fifth test (init_invalid_credentials_raises_api_error) hits the OAuth
endpoint with deliberately-wrong credentials; it requires only network
connectivity — no valid service-account credentials are needed.

If real service-account credentials are available (DX-0079–0084 tests), set:
    PINECONE_CLIENT_ID=<client-id>
    PINECONE_CLIENT_SECRET=<client-secret>

and the admin_credentials fixture (used by future cred-gated tests) will
not skip.
"""

from __future__ import annotations

import os

import pytest

from pinecone import Admin, PineconeValueError
from pinecone.errors import ApiError


@pytest.fixture(scope="module")
def admin_credentials() -> tuple[str, str]:
    """Provide Admin service-account credentials from environment.

    Skips the entire module when both env vars are absent, allowing the
    cred-gated admin tests (DX-0079–0084) to build on this fixture once
    credentials are provisioned.
    """
    client_id = os.getenv("PINECONE_CLIENT_ID")
    client_secret = os.getenv("PINECONE_CLIENT_SECRET")
    if not client_id or not client_secret:
        pytest.skip("PINECONE_CLIENT_ID and PINECONE_CLIENT_SECRET not set")
    return client_id, client_secret


# ---------------------------------------------------------------------------
# init_validation — no real creds required
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_admin_init_validation_missing_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin() raises PineconeValueError mentioning 'client_id' when client_id is absent.

    Env vars are cleared so the Admin.__init__ fallback path also finds nothing.
    Execution exits before any OAuth request is made.
    """
    monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
    monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)

    with pytest.raises(PineconeValueError, match="client_id"):
        Admin(client_secret="anything")


@pytest.mark.integration
def test_admin_init_validation_missing_client_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin() raises PineconeValueError mentioning 'client_secret' when client_secret is absent.

    Env vars are cleared so the Admin.__init__ fallback path also finds nothing.
    Execution exits before any OAuth request is made.
    """
    monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
    monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)

    with pytest.raises(PineconeValueError, match="client_secret"):
        Admin(client_id="anything")


@pytest.mark.integration
def test_admin_init_validation_whitespace_only_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin() raises PineconeValueError for a whitespace-only client_id.

    Env vars are cleared to prevent the fallback from masking the kwarg value.
    Execution exits before any OAuth request is made.
    """
    monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
    monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)

    with pytest.raises(PineconeValueError, match="client_id"):
        Admin(client_id="   ", client_secret="x")


@pytest.mark.integration
def test_admin_init_validation_whitespace_only_client_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admin() raises PineconeValueError for a whitespace-only client_secret.

    Env vars are cleared to prevent the fallback from masking the kwarg value.
    Execution exits before any OAuth request is made.
    """
    monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
    monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)

    with pytest.raises(PineconeValueError, match="client_secret"):
        Admin(client_id="x", client_secret="   ")


@pytest.mark.integration
def test_admin_init_invalid_credentials_raises_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin() raises ApiError when the OAuth endpoint rejects the supplied credentials.

    Uses deliberately-wrong (nonexistent) client_id and client_secret so the
    OAuth server returns a 4xx error.  Requires network connectivity but does
    NOT require valid service-account credentials.

    Env vars are cleared so the Admin.__init__ fallback path does not pick up
    real credentials from the runner environment.
    """
    monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
    monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)

    with pytest.raises(ApiError):
        Admin(client_id="nonexistent", client_secret="nonexistent")
