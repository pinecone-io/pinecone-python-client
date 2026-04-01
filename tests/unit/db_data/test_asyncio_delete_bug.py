"""Reproduction test for GitHub issue #564.

The async delete path fails when the server returns an empty response body
for the delete endpoint. In the user's report (v8.0.0), this manifested as:
    AttributeError: 'str' object has no attribute '_response_info'

On current main (before the fix), the deserialization fails earlier with
PineconeApiTypeError because the empty string can't be converted to dict.

The root cause is the same: an empty response body from the delete endpoint
is not handled gracefully. The fix returns None from the deserializer when
the response body is empty.
"""

import pytest
from unittest.mock import AsyncMock, patch

from pinecone.openapi_support.rest_utils import RESTResponse
from pinecone.openapi_support.asyncio_api_client import AsyncioApiClient
from pinecone.config.openapi_configuration import Configuration


def _make_client_and_response(body: bytes):
    config = Configuration(host="https://test.pinecone.io", api_key={"ApiKeyAuth": "test-key"})
    client = AsyncioApiClient(configuration=config)
    response = RESTResponse(
        status=200,
        data=body,
        headers={"content-type": "application/json"},
        reason="OK",
    )
    return client, response


async def _call_delete(client, response, check_type=True):
    with patch.object(client, "request", new_callable=AsyncMock, return_value=response):
        return await client.call_api(
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
            _check_type=check_type,
            collection_formats={},
        )


class TestAsyncioDeleteBug:
    """Reproduce issue #564: asyncio delete fails with empty response body."""

    @pytest.mark.asyncio
    async def test_delete_empty_body_returns_none(self):
        """When the server returns an empty body (b'') for delete, the async
        client should return None instead of crashing."""
        client, response = _make_client_and_response(b"")
        try:
            result = await _call_delete(client, response)
            assert result is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_delete_with_empty_json_object_succeeds(self):
        """When delete returns '{}', the async client handles it fine."""
        client, response = _make_client_and_response(b"{}")
        try:
            result = await _call_delete(client, response)
            assert isinstance(result, dict)
        finally:
            await client.close()
