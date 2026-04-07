"""Unit tests for AsyncAssistants namespace — create, describe, list, update."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.async_client.assistants import (
    _CREATE_POLL_INTERVAL_SECONDS,
    AsyncAssistants,
)
from pinecone.errors.exceptions import NotFoundError, PineconeTimeoutError, PineconeValueError
from pinecone.models.assistant.list import ListAssistantsResponse
from pinecone.models.assistant.model import AssistantModel
from tests.factories import make_assistant_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture()
def async_assistants() -> AsyncAssistants:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncAssistants(config=config)


# ---------------------------------------------------------------------------
# create() — validation
# ---------------------------------------------------------------------------


def test_create_assistant_region_validation(async_assistants: AsyncAssistants) -> None:
    """Invalid region raises PineconeValueError before any HTTP call."""
    with pytest.raises(PineconeValueError, match="region") as exc_info:
        # Run sync since validation happens before any await
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            async_assistants.create(name="test-assistant", region="ap-southeast-1")
        )
    assert "ap-southeast-1" in str(exc_info.value)


def test_create_assistant_region_case_sensitive(async_assistants: AsyncAssistants) -> None:
    """Uppercase 'US' and 'EU' are rejected — validation is case-sensitive."""
    import asyncio

    with pytest.raises(PineconeValueError, match="region"):
        asyncio.get_event_loop().run_until_complete(
            async_assistants.create(name="test-assistant", region="US")
        )

    with pytest.raises(PineconeValueError, match="region"):
        asyncio.get_event_loop().run_until_complete(
            async_assistants.create(name="test-assistant", region="EU")
        )


# ---------------------------------------------------------------------------
# create() — defaults
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_assistant_defaults(async_assistants: AsyncAssistants) -> None:
    """Default region is 'us', metadata is {}, instructions is None."""
    route = respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = await async_assistants.create(name="test-assistant")

    assert isinstance(result, AssistantModel)
    assert result.name == "test-assistant"

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["region"] == "us"
    assert body["metadata"] == {}
    assert body["instructions"] is None


# ---------------------------------------------------------------------------
# create() — success with immediate return (timeout=-1)
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_assistant_immediate_return(async_assistants: AsyncAssistants) -> None:
    """timeout=-1 returns immediately without polling."""
    route = respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    result = await async_assistants.create(name="test-assistant", timeout=-1)

    assert isinstance(result, AssistantModel)
    assert result.status == "Initializing"
    assert route.call_count == 1


@respx.mock
async def test_create_assistant_with_all_params(async_assistants: AsyncAssistants) -> None:
    """Create with instructions, metadata, and region sends correct body."""
    route = respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    result = await async_assistants.create(
        name="research-bot",
        instructions="You are a research assistant.",
        metadata={"team": "engineering", "version": "1"},
        region="eu",
        timeout=-1,
    )

    assert isinstance(result, AssistantModel)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["name"] == "research-bot"
    assert body["instructions"] == "You are a research assistant."
    assert body["metadata"] == {"team": "engineering", "version": "1"}
    assert body["region"] == "eu"


# ---------------------------------------------------------------------------
# create() — polling
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_polls_until_ready(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """Polling loop calls GET until status is 'Ready'."""
    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    poll_route = respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Ready")),
        ]
    )

    result = await async_assistants.create(name="test-assistant")

    assert result.status == "Ready"
    assert poll_route.call_count == 3


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_polls_with_correct_interval(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """Polling sleeps with the correct interval between polls."""
    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    await async_assistants.create(name="test-assistant")

    from unittest.mock import call

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_CREATE_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# create() — timeout
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.time.monotonic")
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_timeout_raises(
    mock_sleep: object, mock_monotonic: object, async_assistants: AsyncAssistants
) -> None:
    """Exceeding timeout raises PineconeTimeoutError with helpful message."""
    mock_monotonic.side_effect = [0.0, 6.0]  # type: ignore[union-attr]

    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    with pytest.raises(PineconeTimeoutError, match="not ready") as exc_info:
        await async_assistants.create(name="test-assistant", timeout=5)

    assert "describe_assistant" in str(exc_info.value)


@respx.mock
@patch("pinecone.async_client.assistants.time.monotonic")
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_timeout_zero_polls_once(
    mock_sleep: object, mock_monotonic: object, async_assistants: AsyncAssistants
) -> None:
    """timeout=0 polls once, then raises if not ready."""
    mock_monotonic.side_effect = [0.0, 0.0, 0.1]  # type: ignore[union-attr]

    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    poll_route = respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
        ]
    )

    with pytest.raises(PineconeTimeoutError):
        await async_assistants.create(name="test-assistant", timeout=0)

    assert poll_route.call_count >= 1


# ---------------------------------------------------------------------------
# create() — status check is case-sensitive
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.time.monotonic")
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_status_case_sensitive(
    mock_sleep: object, mock_monotonic: object, async_assistants: AsyncAssistants
) -> None:
    """Status check uses exact 'Ready' — 'ready' or 'READY' does not match."""
    mock_monotonic.side_effect = [0.0, 6.0]  # type: ignore[union-attr]

    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="ready")),
    )

    with pytest.raises(PineconeTimeoutError):
        await async_assistants.create(name="test-assistant", timeout=5)


# ---------------------------------------------------------------------------
# create() — indefinite polling (timeout=None)
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.async_client.assistants.asyncio.sleep")
async def test_create_assistant_polls_indefinitely_when_no_timeout(
    mock_sleep: object, async_assistants: AsyncAssistants
) -> None:
    """When timeout is None, polling continues until Ready with no deadline."""
    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Initializing")),
            httpx.Response(200, json=make_assistant_response(status="Ready")),
        ]
    )

    result = await async_assistants.create(name="test-assistant", timeout=None)

    assert result.status == "Ready"


# ---------------------------------------------------------------------------
# create() — valid regions accepted
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_assistant_accepts_us_region(async_assistants: AsyncAssistants) -> None:
    """Region 'us' is accepted."""
    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = await async_assistants.create(name="test-assistant", region="us")
    assert isinstance(result, AssistantModel)


@respx.mock
async def test_create_assistant_accepts_eu_region(async_assistants: AsyncAssistants) -> None:
    """Region 'eu' is accepted."""
    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = await async_assistants.create(name="test-assistant", region="eu")
    assert isinstance(result, AssistantModel)


# ---------------------------------------------------------------------------
# describe() — success
# ---------------------------------------------------------------------------


@respx.mock
async def test_describe_assistant(async_assistants: AsyncAssistants) -> None:
    """describe() sends GET /assistants/{name} and returns AssistantModel."""
    response_data = make_assistant_response(
        name="my-assistant",
        status="Ready",
        instructions="Be helpful.",
        metadata={"team": "ml"},
        host="my-assistant-abc.svc.pinecone.io",
    )
    route = respx.get(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=response_data),
    )

    result = await async_assistants.describe(name="my-assistant")

    assert isinstance(result, AssistantModel)
    assert result.name == "my-assistant"
    assert result.status == "Ready"
    assert result.instructions == "Be helpful."
    assert result.metadata == {"team": "ml"}
    assert result.host == "my-assistant-abc.svc.pinecone.io"
    assert result.created_at == "2025-01-15T12:00:00Z"
    assert result.updated_at == "2025-01-15T12:00:00Z"
    assert route.call_count == 1


@respx.mock
async def test_describe_assistant_not_found(async_assistants: AsyncAssistants) -> None:
    """describe() lets 404 errors propagate from the HTTP client."""
    respx.get(f"{BASE_URL}/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        await async_assistants.describe(name="nonexistent")


@respx.mock
async def test_describe_assistant_minimal_response(async_assistants: AsyncAssistants) -> None:
    """describe() handles response with optional fields absent."""
    response_data = make_assistant_response(
        name="minimal",
        metadata=None,
        instructions=None,
        host=None,
    )
    respx.get(f"{BASE_URL}/assistants/minimal").mock(
        return_value=httpx.Response(200, json=response_data),
    )

    result = await async_assistants.describe(name="minimal")

    assert result.name == "minimal"
    assert result.metadata is None
    assert result.instructions is None
    assert result.host is None


# ---------------------------------------------------------------------------
# list() — auto-pagination
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_assistants_empty(async_assistants: AsyncAssistants) -> None:
    """list() returns empty list when no assistants exist."""
    respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    result = await async_assistants.list()

    assert result == []


@respx.mock
async def test_list_assistants_single_page(async_assistants: AsyncAssistants) -> None:
    """list() returns all assistants from a single-page response."""
    respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [
                    make_assistant_response(name="a1"),
                    make_assistant_response(name="a2"),
                ],
            },
        ),
    )

    result = await async_assistants.list()

    assert len(result) == 2
    assert result[0].name == "a1"
    assert result[1].name == "a2"
    assert all(isinstance(a, AssistantModel) for a in result)


@respx.mock
async def test_list_assistants_multi_page(async_assistants: AsyncAssistants) -> None:
    """list() auto-paginates through multiple pages."""
    respx.get(f"{BASE_URL}/assistants").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "assistants": [make_assistant_response(name="a1")],
                    "next": "token-page2",
                },
            ),
            httpx.Response(
                200,
                json={
                    "assistants": [make_assistant_response(name="a2")],
                    "next": "token-page3",
                },
            ),
            httpx.Response(
                200,
                json={
                    "assistants": [make_assistant_response(name="a3")],
                },
            ),
        ]
    )

    result = await async_assistants.list()

    assert len(result) == 3
    assert [a.name for a in result] == ["a1", "a2", "a3"]


# ---------------------------------------------------------------------------
# list_page() — single page with pagination control
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_assistants_page(async_assistants: AsyncAssistants) -> None:
    """list_page() returns single page with next token."""
    route = respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [make_assistant_response(name="a1")],
                "next": "token-next",
            },
        ),
    )

    result = await async_assistants.list_page()

    assert isinstance(result, ListAssistantsResponse)
    assert len(result.assistants) == 1
    assert result.assistants[0].name == "a1"
    assert result.next == "token-next"
    assert route.call_count == 1


@respx.mock
async def test_list_assistants_page_last_page(async_assistants: AsyncAssistants) -> None:
    """list_page() returns no next token on the last page."""
    respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [make_assistant_response(name="a1")],
            },
        ),
    )

    result = await async_assistants.list_page()

    assert result.next is None


@respx.mock
async def test_list_assistants_page_with_page_size(async_assistants: AsyncAssistants) -> None:
    """list_page() sends pageSize query param when provided."""
    route = respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    await async_assistants.list_page(page_size=5)

    request = route.calls.last.request
    assert "pageSize=5" in str(request.url)


@respx.mock
async def test_list_assistants_page_with_pagination_token(
    async_assistants: AsyncAssistants,
) -> None:
    """list_page() sends paginationToken query param when provided."""
    route = respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    await async_assistants.list_page(pagination_token="abc123")

    request = route.calls.last.request
    assert "paginationToken=abc123" in str(request.url)


@respx.mock
async def test_list_assistants_page_omits_none_params(
    async_assistants: AsyncAssistants,
) -> None:
    """list_page() does not send params that are None."""
    route = respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    await async_assistants.list_page()

    request = route.calls.last.request
    assert "pageSize" not in str(request.url)
    assert "paginationToken" not in str(request.url)


# ---------------------------------------------------------------------------
# update() — success
# ---------------------------------------------------------------------------


@respx.mock
async def test_update_assistant_instructions(async_assistants: AsyncAssistants) -> None:
    """update() sends PATCH /assistants/{name} with instructions."""
    updated_response = make_assistant_response(
        name="my-assistant",
        instructions="Updated instructions.",
    )
    route = respx.patch(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = await async_assistants.update(
        name="my-assistant", instructions="Updated instructions."
    )

    assert isinstance(result, AssistantModel)
    assert result.name == "my-assistant"
    assert result.instructions == "Updated instructions."
    assert route.call_count == 1

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"instructions": "Updated instructions."}


@respx.mock
async def test_update_assistant_metadata(async_assistants: AsyncAssistants) -> None:
    """update() sends PATCH with metadata."""
    new_metadata = {"team": "ml", "version": "2"}
    updated_response = make_assistant_response(
        name="my-assistant",
        metadata=new_metadata,
    )
    route = respx.patch(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = await async_assistants.update(name="my-assistant", metadata=new_metadata)

    assert isinstance(result, AssistantModel)
    assert result.metadata == new_metadata
    assert route.call_count == 1

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"metadata": new_metadata}


@respx.mock
async def test_update_assistant_both_fields(async_assistants: AsyncAssistants) -> None:
    """update() sends both instructions and metadata when provided."""
    updated_response = make_assistant_response(
        name="my-assistant",
        instructions="New instructions.",
        metadata={"env": "prod"},
    )
    route = respx.patch(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = await async_assistants.update(
        name="my-assistant",
        instructions="New instructions.",
        metadata={"env": "prod"},
    )

    assert result.instructions == "New instructions."
    assert result.metadata == {"env": "prod"}

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"instructions": "New instructions.", "metadata": {"env": "prod"}}


@respx.mock
async def test_update_assistant_omits_none_fields(async_assistants: AsyncAssistants) -> None:
    """update() only includes provided fields in the request body."""
    updated_response = make_assistant_response(name="my-assistant")
    route = respx.patch(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    await async_assistants.update(name="my-assistant", instructions="Only this.")

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "metadata" not in body
    assert body == {"instructions": "Only this."}


@respx.mock
async def test_update_assistant_not_found(async_assistants: AsyncAssistants) -> None:
    """update() lets 404 errors propagate from the HTTP client."""
    respx.patch(f"{BASE_URL}/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        await async_assistants.update(name="nonexistent", instructions="test")


# ---------------------------------------------------------------------------
# delete() — success
# ---------------------------------------------------------------------------


@respx.mock
async def test_delete_assistant(async_assistants: AsyncAssistants) -> None:
    """delete() sends DELETE /assistants/{name} and returns None."""
    route = respx.delete(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )

    result = await async_assistants.delete(name="my-assistant")

    assert result is None
    assert route.call_count == 1

    request = route.calls.last.request
    assert request.method == "DELETE"
    assert str(request.url) == f"{BASE_URL}/assistants/my-assistant"


@respx.mock
async def test_delete_assistant_not_found(async_assistants: AsyncAssistants) -> None:
    """delete() lets 404 errors propagate from the HTTP client."""
    respx.delete(f"{BASE_URL}/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        await async_assistants.delete(name="nonexistent")


# ---------------------------------------------------------------------------
# repr()
# ---------------------------------------------------------------------------


def test_repr(async_assistants: AsyncAssistants) -> None:
    assert repr(async_assistants) == "AsyncAssistants()"


# ---------------------------------------------------------------------------
# AsyncPinecone.assistants property
# ---------------------------------------------------------------------------


def test_async_pinecone_assistants_property() -> None:
    from pinecone.async_client.pinecone import AsyncPinecone

    pc = AsyncPinecone(api_key="test-key")
    assistants = pc.assistants
    assert isinstance(assistants, AsyncAssistants)
    # Verify lazy caching — same instance returned
    assert pc.assistants is assistants
