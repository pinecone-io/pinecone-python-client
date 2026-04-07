"""Unit tests for the Admin client constructor and OAuth token flow."""

from __future__ import annotations

from typing import Any

import pytest
import respx
from httpx import Response

from pinecone.admin.admin import _OAUTH_URL, Admin
from pinecone.errors.exceptions import ApiError, ValidationError


def _token_response(token: str = "test-access-token") -> dict[str, Any]:
    """Return a valid OAuth token response payload."""
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": 1800,
    }


class TestAdminValidation:
    """Test validation of client_id and client_secret."""

    def test_admin_requires_client_id(self) -> None:
        with pytest.raises(ValidationError, match="client_id"):
            Admin(client_secret="secret")

    def test_admin_requires_client_secret(self) -> None:
        with pytest.raises(ValidationError, match="client_secret"):
            Admin(client_id="id")

    def test_admin_rejects_empty_client_id(self) -> None:
        with pytest.raises(ValidationError, match="client_id"):
            Admin(client_id="", client_secret="secret")

    def test_admin_rejects_empty_client_secret(self) -> None:
        with pytest.raises(ValidationError, match="client_secret"):
            Admin(client_id="id", client_secret="")

    def test_admin_rejects_whitespace_client_id(self) -> None:
        with pytest.raises(ValidationError, match="client_id"):
            Admin(client_id="   ", client_secret="secret")

    def test_admin_rejects_whitespace_client_secret(self) -> None:
        with pytest.raises(ValidationError, match="client_secret"):
            Admin(client_id="id", client_secret="   ")

    def test_admin_no_env_fallback_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
        monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)
        with pytest.raises(ValidationError, match="client_id"):
            Admin()


class TestAdminEnvVarFallback:
    """Test environment variable fallback for credentials."""

    @respx.mock
    def test_admin_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CLIENT_ID", "env-client-id")
        monkeypatch.setenv("PINECONE_CLIENT_SECRET", "env-client-secret")

        respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response())
        )

        admin = Admin()
        assert admin._http is not None
        admin.close()

    @respx.mock
    def test_admin_explicit_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_CLIENT_ID", "env-id")
        monkeypatch.setenv("PINECONE_CLIENT_SECRET", "env-secret")

        route = respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response())
        )

        admin = Admin(client_id="explicit-id", client_secret="explicit-secret")
        admin.close()

        # Verify the explicit credentials were sent, not the env vars
        request = route.calls.last.request
        import orjson

        body = orjson.loads(request.content)
        assert body["client_id"] == "explicit-id"
        assert body["client_secret"] == "explicit-secret"


class TestAdminTokenFetch:
    """Test the OAuth token exchange flow."""

    @respx.mock
    def test_admin_fetches_bearer_token(self) -> None:
        route = respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response("my-bearer-token"))
        )

        admin = Admin(client_id="test-id", client_secret="test-secret")

        # Verify the OAuth request was made with correct body
        assert route.called
        request = route.calls.last.request
        import orjson

        body = orjson.loads(request.content)
        assert body["client_id"] == "test-id"
        assert body["client_secret"] == "test-secret"
        assert body["grant_type"] == "client_credentials"
        assert body["audience"] == "https://api.pinecone.io/"

        # Verify the Bearer token is used in the HTTPClient headers
        auth_header = admin._http._headers.get("Authorization")
        assert auth_header == "Bearer my-bearer-token"

        admin.close()

    @respx.mock
    def test_admin_token_request_includes_api_version(self) -> None:
        route = respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response())
        )

        admin = Admin(client_id="test-id", client_secret="test-secret")

        request = route.calls.last.request
        assert request.headers["X-Pinecone-Api-Version"] == "2025-10"

        admin.close()

    @respx.mock
    def test_admin_token_fetch_401_raises_api_error(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(
                401,
                json={"error": "access_denied", "error_description": "Unauthorized"},
            )
        )

        with pytest.raises(ApiError, match="Unauthorized") as exc_info:
            Admin(client_id="bad-id", client_secret="bad-secret")

        assert exc_info.value.status_code == 401

    @respx.mock
    def test_admin_token_fetch_403_raises_api_error(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(
                403,
                json={"error": "forbidden", "error_description": "Forbidden"},
            )
        )

        with pytest.raises(ApiError, match="Forbidden") as exc_info:
            Admin(client_id="bad-id", client_secret="bad-secret")

        assert exc_info.value.status_code == 403

    @respx.mock
    def test_admin_token_fetch_missing_access_token(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json={"token_type": "Bearer", "expires_in": 1800})
        )

        with pytest.raises(ApiError, match="missing access_token"):
            Admin(client_id="test-id", client_secret="test-secret")

    @respx.mock
    def test_admin_token_fetch_error_without_description(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(400, json={"error": "invalid_request"})
        )

        with pytest.raises(ApiError, match="invalid_request") as exc_info:
            Admin(client_id="test-id", client_secret="test-secret")

        assert exc_info.value.status_code == 400


class TestAdminHeaders:
    """Test that the Admin client configures correct headers."""

    @respx.mock
    def test_admin_sets_api_version_header(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response())
        )

        admin = Admin(client_id="test-id", client_secret="test-secret")
        assert admin._http._headers["X-Pinecone-Api-Version"] == "2025-10"
        admin.close()

    @respx.mock
    def test_admin_additional_headers(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response())
        )

        admin = Admin(
            client_id="test-id",
            client_secret="test-secret",
            additional_headers={"X-Custom": "value"},
        )
        assert admin._http._headers["X-Custom"] == "value"
        assert admin._http._headers["Authorization"] == "Bearer test-access-token"
        admin.close()


class TestAdminApiKeyNotLeaked:
    """Test that the Admin client does not leak data-plane API keys."""

    @respx.mock
    def test_admin_does_not_include_api_key_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When PINECONE_API_KEY is set, Admin must NOT send it as Api-Key."""
        monkeypatch.setenv("PINECONE_API_KEY", "data-plane-key-12345")

        respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response())
        )

        admin = Admin(client_id="test-id", client_secret="test-secret")
        assert "Api-Key" not in admin._http._headers
        assert admin._http._headers["Authorization"] == "Bearer test-access-token"
        admin.close()

    @respx.mock
    def test_admin_api_key_empty_without_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without PINECONE_API_KEY env var, Api-Key header is still absent."""
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)

        respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response())
        )

        admin = Admin(client_id="test-id", client_secret="test-secret")
        assert "Api-Key" not in admin._http._headers
        admin.close()


class TestAdminContextManager:
    """Test context manager support."""

    @respx.mock
    def test_admin_context_manager(self) -> None:
        respx.post(_OAUTH_URL).mock(
            return_value=Response(200, json=_token_response())
        )

        with Admin(client_id="test-id", client_secret="test-secret") as admin:
            assert admin._http is not None
