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
import time

import pytest

from pinecone import Admin, PineconeValueError
from pinecone.errors import ApiError
from pinecone.models.admin.organization import OrganizationList, OrganizationModel


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


# ---------------------------------------------------------------------------
# admin fixture — requires real credentials (PINECONE_CLIENT_ID / SECRET)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def admin(admin_credentials: tuple[str, str]) -> Admin:
    """Construct an authenticated Admin client using real service-account credentials.

    Module-scoped so the OAuth token exchange happens once per test run.
    Skipped automatically when admin_credentials skips (no env vars).
    """
    client_id, client_secret = admin_credentials
    return Admin(client_id=client_id, client_secret=client_secret)


# ---------------------------------------------------------------------------
# organizations — list / describe / update
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_organizations_list_returns_iterable_of_models(admin: Admin) -> None:
    """admin.organizations.list() returns an OrganizationList with at least one org.

    Verifies:
    - Return type is OrganizationList (supports iter and len).
    - At least one organization is present (service accounts always belong to one).
    - Every element is an OrganizationModel with non-empty id and name.
    """
    orgs = admin.organizations.list()

    assert isinstance(orgs, OrganizationList)
    assert len(orgs) >= 1

    for org in orgs:
        assert isinstance(org, OrganizationModel)
        assert isinstance(org.id, str) and org.id
        assert isinstance(org.name, str) and org.name


@pytest.mark.integration
def test_organizations_describe_returns_matching_model(admin: Admin) -> None:
    """admin.organizations.describe() returns a model matching the listed org.

    Takes the first org from list(), calls describe() with its id, and
    verifies that the described model has consistent id and name and that
    all required fields are populated.
    """
    orgs = admin.organizations.list()
    first = orgs[0]

    described = admin.organizations.describe(organization_id=first.id)

    assert isinstance(described, OrganizationModel)
    assert described.id == first.id
    assert described.name == first.name
    assert described.plan
    assert described.payment_status
    assert described.created_at
    assert described.support_tier


@pytest.mark.integration
def test_organizations_update_roundtrips_name(admin: Admin) -> None:
    """admin.organizations.update() persists the new name and is reversible.

    Saves the original name, renames to a timestamped variant, asserts the
    returned model reflects the new name, re-describes to confirm the change
    persisted on the server, then restores the original name in a finally
    block to leave org state clean.
    """
    orgs = admin.organizations.list()
    first = orgs[0]
    original_name = first.name
    new_name = f"{original_name}-test-{int(time.time())}"

    try:
        updated = admin.organizations.update(organization_id=first.id, name=new_name)
        assert isinstance(updated, OrganizationModel)
        assert updated.name == new_name

        # Re-describe to confirm server persisted the change.
        described = admin.organizations.describe(organization_id=first.id)
        assert described.name == new_name
    finally:
        admin.organizations.update(organization_id=first.id, name=original_name)
