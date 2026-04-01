"""Reproduction test for GitHub issue #564.

The async delete path fails when the server returns an empty response body
for the delete endpoint. In the user's report (v8.0.0), this manifested as:
    AttributeError: 'str' object has no attribute '_response_info'

The root cause was that the deserializer returned a raw empty string when
the response body couldn't be parsed as JSON, and downstream code tried
to set attributes on that string.

The fix returns None early from the deserializer when response data is
empty or whitespace-only.
"""

import pytest
from unittest.mock import AsyncMock, patch

from pinecone.openapi_support.rest_utils import RESTResponse
from pinecone.openapi_support.asyncio_api_client import AsyncioApiClient
from pinecone.openapi_support.api_client import ApiClient
from pinecone.config.openapi_configuration import Configuration


def _make_config():
    return Configuration(host="https://test.pinecone.io", api_key={"ApiKeyAuth": "test-key"})


def _make_response(body: bytes):
    return RESTResponse(
        status=200,
        data=body,
        headers={"content-type": "application/json"},
        reason="OK",
    )


_CALL_API_KWARGS = dict(
    resource_path="/vectors/delete",
    method="POST",
    path_params={},
    query_params=[],
    header_params={"Content-Type": "application/json"},
    body={"deleteAll": True, "namespace": "test-namespace"},
    response_type=(dict,),
    auth_settings=["ApiKeyAuth"],
    _return_http_data_only=True,
    _preload_content=True,
    _request_timeout=None,
    _host=None,
    _check_type=True,
    collection_formats={},
)


class TestAsyncioDeleteEmptyBody:
    """Async client: empty/whitespace response bodies should return None."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [b"", b" ", b"\n", b"  \n  "])
    async def test_empty_or_whitespace_body_returns_none(self, body):
        client = AsyncioApiClient(configuration=_make_config())
        try:
            with patch.object(
                client, "request", new_callable=AsyncMock, return_value=_make_response(body)
            ):
                result = await client.call_api(**_CALL_API_KWARGS)
                assert result is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_empty_json_object_succeeds(self):
        client = AsyncioApiClient(configuration=_make_config())
        try:
            with patch.object(
                client, "request", new_callable=AsyncMock, return_value=_make_response(b"{}")
            ):
                result = await client.call_api(**_CALL_API_KWARGS)
                assert isinstance(result, dict)
        finally:
            await client.close()


class TestSyncDeleteEmptyBody:
    """Sync client: empty/whitespace response bodies should return None."""

    @pytest.mark.parametrize("body", [b"", b" ", b"\n", b"  \n  "])
    def test_empty_or_whitespace_body_returns_none(self, body):
        client = ApiClient(configuration=_make_config())
        with patch.object(client, "request", return_value=_make_response(body)):
            result = client.call_api(**_CALL_API_KWARGS)
            assert result is None

    def test_empty_json_object_succeeds(self):
        client = ApiClient(configuration=_make_config())
        with patch.object(client, "request", return_value=_make_response(b"{}")):
            result = client.call_api(**_CALL_API_KWARGS)
            assert isinstance(result, dict)
