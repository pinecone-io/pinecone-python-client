"""Property-based characterization of Admin(additional_headers=...) precedence.

The documented rule, pinned here: ``additional_headers`` is merged last, so a
key named ``Authorization`` or ``X-Pinecone-Api-Version``, in any capitalization,
wins over the Bearer token and the SDK's version constant — that is the
deliberate escape hatch, and exactly one value of each is sent. Any other key
is carried through without displacing them.
"""

from __future__ import annotations

import string
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone._internal.constants import API_VERSION_HEADER
from pinecone._internal.http_client import HTTPClient
from pinecone.admin.admin import _OAUTH_URL, Admin

_TOKEN: dict[str, Any] = {
    "access_token": "test-access-token",
    "token_type": "Bearer",
    "expires_in": 1800,
}
_EXPECTED_AUTHORIZATION = "Bearer test-access-token"
_PROTECTED = {"authorization", API_VERSION_HEADER.lower()}

_HEADER_NAMES = st.text(
    alphabet=string.ascii_letters + string.digits + "-", min_size=1, max_size=24
).filter(lambda name: name.lower() not in _PROTECTED)
_HEADER_VALUES = st.text(alphabet=string.ascii_letters + string.digits + " -_.", max_size=32)


@contextmanager
def _admin(**kwargs: Any) -> Iterator[Admin]:
    with respx.mock(assert_all_called=False) as router:
        router.post(_OAUTH_URL).mock(return_value=httpx.Response(200, json=_TOKEN))
        client = Admin(client_id="test-id", client_secret="test-secret", **kwargs)
        try:
            yield client
        finally:
            client.close()


@given(extra=st.dictionaries(_HEADER_NAMES, _HEADER_VALUES, max_size=8))
@settings(max_examples=50, deadline=None)
def test_additional_headers_never_displace_auth_or_version(extra: dict[str, str]) -> None:
    with _admin(additional_headers=extra) as admin:
        headers = admin._http._headers
        assert headers["Authorization"] == _EXPECTED_AUTHORIZATION
        assert headers[API_VERSION_HEADER] == "2026-07"
        surviving = {name.lower(): (name, value) for name, value in extra.items()}
        for name, value in surviving.values():
            assert headers[name] == value


def test_exact_authorization_key_wins() -> None:
    with _admin(additional_headers={"Authorization": "Bearer caller-supplied"}) as admin:
        assert admin._http._headers["Authorization"] == "Bearer caller-supplied"
        assert admin._http._headers[API_VERSION_HEADER] == "2026-07"


def test_exact_api_version_key_wins() -> None:
    with _admin(additional_headers={API_VERSION_HEADER: "2025-10"}) as admin:
        assert admin._http._headers[API_VERSION_HEADER] == "2025-10"
        assert admin._http._headers["Authorization"] == _EXPECTED_AUTHORIZATION


def test_differently_cased_keys_replace_the_canonical_headers() -> None:
    variants = {"authorization": "Bearer lowercase", "x-pinecone-api-version": "1999-01"}
    with _admin(additional_headers=variants) as admin:
        headers = admin._http._headers
        assert headers["authorization"] == "Bearer lowercase"
        assert headers["x-pinecone-api-version"] == "1999-01"
        assert "Authorization" not in headers
        assert API_VERSION_HEADER not in headers


def test_differently_cased_authorization_opts_out_of_token_refresh() -> None:
    with _admin(additional_headers={"authorization": "Bearer lowercase"}) as admin:
        assert type(admin._http) is HTTPClient
