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
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.pagination import AsyncPaginator, Page
from tests.factories import make_assistant_file_response, make_assistant_response

BASE_URL = "https://api.test.pinecone.io"
DATA_PLANE_HOST = "test-assistant-abc123.svc.pinecone.io"
DATA_PLANE_URL = f"https://{DATA_PLANE_HOST}"


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

    assert "pc.assistants.describe" in str(exc_info.value)


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
    """list() returns an AsyncPaginator that yields nothing when no assistants exist."""
    respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    result = async_assistants.list()

    assert isinstance(result, AsyncPaginator)
    assert await result.to_list() == []


@respx.mock
async def test_list_assistants_single_page(async_assistants: AsyncAssistants) -> None:
    """list() returns an AsyncPaginator over all assistants from a single-page response."""
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

    result = await async_assistants.list().to_list()

    assert len(result) == 2
    assert result[0].name == "a1"
    assert result[1].name == "a2"
    assert all(isinstance(a, AssistantModel) for a in result)


@respx.mock
async def test_list_assistants_multi_page(async_assistants: AsyncAssistants) -> None:
    """list() auto-paginates through multiple pages via AsyncPaginator."""
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

    result = await async_assistants.list().to_list()

    assert len(result) == 3
    assert [a.name for a in result] == ["a1", "a2", "a3"]


@respx.mock
async def test_list_assistants_to_list(async_assistants: AsyncAssistants) -> None:
    """list().to_list() collects all assistants into a list."""
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

    items = await async_assistants.list().to_list()

    assert len(items) == 2
    assert all(isinstance(a, AssistantModel) for a in items)


@respx.mock
async def test_list_assistants_iteration(async_assistants: AsyncAssistants) -> None:
    """list() supports direct async for-loop iteration."""
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

    names = [a.name async for a in async_assistants.list()]

    assert names == ["a1", "a2"]


@respx.mock
async def test_list_assistants_with_limit(async_assistants: AsyncAssistants) -> None:
    """list(limit=N) yields at most N items across all pages."""
    respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [
                    make_assistant_response(name="a1"),
                    make_assistant_response(name="a2"),
                    make_assistant_response(name="a3"),
                ],
                "next": "more",
            },
        ),
    )

    result = await async_assistants.list(limit=2).to_list()

    assert len(result) == 2
    assert result[0].name == "a1"
    assert result[1].name == "a2"


@respx.mock
async def test_list_assistants_pages(async_assistants: AsyncAssistants) -> None:
    """list().pages() yields Page objects with items and has_more."""
    respx.get(f"{BASE_URL}/assistants").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "assistants": [make_assistant_response(name="a1")],
                    "next": "token-next",
                },
            ),
            httpx.Response(
                200,
                json={
                    "assistants": [make_assistant_response(name="a2")],
                },
            ),
        ]
    )

    pages = [p async for p in async_assistants.list().pages()]

    assert len(pages) == 2
    assert pages[0].has_more is True
    assert pages[0].items[0].name == "a1"
    assert pages[1].has_more is False
    assert pages[1].items[0].name == "a2"
    for page in pages:
        assert isinstance(page, Page)


@respx.mock
async def test_list_assistants_with_pagination_token(async_assistants: AsyncAssistants) -> None:
    """list(pagination_token=...) starts from the given token."""
    route = respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(
            200,
            json={"assistants": [make_assistant_response(name="a2")]},
        ),
    )

    result = await async_assistants.list(pagination_token="tok-page2").to_list()

    assert len(result) == 1
    assert result[0].name == "a2"
    request = route.calls.last.request
    assert "paginationToken=tok-page2" in str(request.url)


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


# ---------------------------------------------------------------------------
# __init__ — data-plane and eval HTTP infrastructure
# ---------------------------------------------------------------------------


def test_init_creates_eval_http(async_assistants: AsyncAssistants) -> None:
    """Constructor creates _eval_http targeting ASSISTANT_EVALUATION_BASE_URL."""
    from pinecone._internal.constants import ASSISTANT_EVALUATION_BASE_URL
    from pinecone._internal.http_client import AsyncHTTPClient

    assert hasattr(async_assistants, "_eval_http")
    assert isinstance(async_assistants._eval_http, AsyncHTTPClient)
    # _eval_http should be configured with the evaluation base URL
    assert async_assistants._eval_http._config.host == ASSISTANT_EVALUATION_BASE_URL


def test_init_creates_empty_data_plane_clients(async_assistants: AsyncAssistants) -> None:
    """Constructor initializes _data_plane_clients as an empty dict."""
    assert hasattr(async_assistants, "_data_plane_clients")
    assert isinstance(async_assistants._data_plane_clients, dict)
    assert len(async_assistants._data_plane_clients) == 0


# ---------------------------------------------------------------------------
# close() — eval_http and data-plane clients
# ---------------------------------------------------------------------------


@respx.mock
async def test_close_closes_eval_http(async_assistants: AsyncAssistants) -> None:
    """close() closes _eval_http in addition to _http."""
    from unittest.mock import AsyncMock

    async_assistants._eval_http.close = AsyncMock()  # type: ignore[method-assign]

    await async_assistants.close()

    async_assistants._eval_http.close.assert_called_once()  # type: ignore[union-attr]


@respx.mock
async def test_close_closes_data_plane_clients(async_assistants: AsyncAssistants) -> None:
    """close() closes all cached data-plane clients and clears the cache."""
    from unittest.mock import AsyncMock

    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.http_client import AsyncHTTPClient

    # Inject a fake data-plane client into the cache
    fake_client = AsyncHTTPClient(
        PineconeConfig(api_key="test-key", host="https://fake.host"), "2025-10"
    )
    fake_client.close = AsyncMock()  # type: ignore[method-assign]
    async_assistants._data_plane_clients["my-assistant"] = fake_client

    await async_assistants.close()

    fake_client.close.assert_called_once()  # type: ignore[union-attr]
    assert len(async_assistants._data_plane_clients) == 0


# ---------------------------------------------------------------------------
# _data_plane_http() — caching and error handling
# ---------------------------------------------------------------------------


@respx.mock
async def test_data_plane_http_returns_client_for_host(
    async_assistants: AsyncAssistants,
) -> None:
    """_data_plane_http() returns an AsyncHTTPClient configured with the assistant's host."""
    from pinecone._internal.http_client import AsyncHTTPClient

    respx.get(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(
                name="my-assistant", host="my-assistant-abc.svc.pinecone.io"
            ),
        ),
    )

    client = await async_assistants._data_plane_http("my-assistant")

    assert isinstance(client, AsyncHTTPClient)
    assert client._config.host == "https://my-assistant-abc.svc.pinecone.io"


@respx.mock
async def test_data_plane_http_caches_client(async_assistants: AsyncAssistants) -> None:
    """_data_plane_http() returns the same client on repeated calls (no extra describe)."""
    describe_route = respx.get(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(
                name="my-assistant", host="my-assistant-abc.svc.pinecone.io"
            ),
        ),
    )

    client1 = await async_assistants._data_plane_http("my-assistant")
    client2 = await async_assistants._data_plane_http("my-assistant")

    assert client1 is client2
    assert describe_route.call_count == 1


@respx.mock
async def test_data_plane_http_different_assistants_get_different_clients(
    async_assistants: AsyncAssistants,
) -> None:
    """_data_plane_http() caches separately per assistant name."""
    respx.get(f"{BASE_URL}/assistants/assistant-a").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(name="assistant-a", host="host-a.svc.pinecone.io"),
        ),
    )
    respx.get(f"{BASE_URL}/assistants/assistant-b").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(name="assistant-b", host="host-b.svc.pinecone.io"),
        ),
    )

    client_a = await async_assistants._data_plane_http("assistant-a")
    client_b = await async_assistants._data_plane_http("assistant-b")

    assert client_a is not client_b
    assert client_a._config.host == "https://host-a.svc.pinecone.io"
    assert client_b._config.host == "https://host-b.svc.pinecone.io"


@respx.mock
async def test_data_plane_http_raises_when_host_is_none(
    async_assistants: AsyncAssistants,
) -> None:
    """_data_plane_http() raises PineconeValueError when assistant has no host."""
    respx.get(f"{BASE_URL}/assistants/no-host-assistant").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(name="no-host-assistant", host=None),
        ),
    )

    with pytest.raises(PineconeValueError, match="no data-plane host"):
        await async_assistants._data_plane_http("no-host-assistant")


@respx.mock
async def test_data_plane_http_raises_when_host_is_empty_string(
    async_assistants: AsyncAssistants,
) -> None:
    """_data_plane_http() raises PineconeValueError when assistant host is empty string."""
    respx.get(f"{BASE_URL}/assistants/empty-host-assistant").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_response(name="empty-host-assistant", host=""),
        ),
    )

    with pytest.raises(PineconeValueError, match="no data-plane host"):
        await async_assistants._data_plane_http("empty-host-assistant")


# ---------------------------------------------------------------------------
# describe_file() — success
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_describe_file_success(async_assistants: AsyncAssistants) -> None:
    """describe_file() sends GET /files/{name}/{id} via data-plane and returns AssistantFileModel."""  # noqa: E501
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response()),
    )

    result = await async_assistants.describe_file(
        assistant_name="test-assistant", file_id="file-abc123"
    )

    assert isinstance(result, AssistantFileModel)
    assert result.id == "file-abc123"
    assert result.name == "test-file.pdf"
    assert route.call_count == 1


@respx.mock
async def test_async_describe_file_without_url(async_assistants: AsyncAssistants) -> None:
    """describe_file() does not send include_url param by default."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response()),
    )

    await async_assistants.describe_file(assistant_name="test-assistant", file_id="file-abc123")

    request = route.calls.last.request
    assert "include_url" not in str(request.url)


@respx.mock
async def test_async_describe_file_with_url(async_assistants: AsyncAssistants) -> None:
    """describe_file(include_url=True) sends include_url=true query param."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_file_response(signed_url="https://storage.example.com/file-abc123"),
        ),
    )

    result = await async_assistants.describe_file(
        assistant_name="test-assistant", file_id="file-abc123", include_url=True
    )

    request = route.calls.last.request
    assert "include_url=true" in str(request.url)
    assert result.signed_url == "https://storage.example.com/file-abc123"


@respx.mock
async def test_async_describe_file_not_found(async_assistants: AsyncAssistants) -> None:
    """describe_file() raises NotFoundError when file does not exist."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        await async_assistants.describe_file(assistant_name="test-assistant", file_id="nonexistent")


# ---------------------------------------------------------------------------
# list_files_page() — single page
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_list_files_page_success(async_assistants: AsyncAssistants) -> None:
    """list_files_page() returns ListFilesResponse with files and next token."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={
                "files": [make_assistant_file_response()],
                "next": "token-next",
            },
        ),
    )

    result = await async_assistants.list_files_page(assistant_name="test-assistant")

    assert isinstance(result, ListFilesResponse)
    assert len(result.files) == 1
    assert result.files[0].id == "file-abc123"
    assert result.next == "token-next"
    assert route.call_count == 1


@respx.mock
async def test_async_list_files_page_last_page(async_assistants: AsyncAssistants) -> None:
    """list_files_page() returns no next token on the last page."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": [make_assistant_file_response()]}),
    )

    result = await async_assistants.list_files_page(assistant_name="test-assistant")

    assert result.next is None


@respx.mock
async def test_async_list_files_page_with_page_size(
    async_assistants: AsyncAssistants,
) -> None:
    """list_files_page() sends pageSize query param when provided."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    await async_assistants.list_files_page(assistant_name="test-assistant", page_size=5)

    request = route.calls.last.request
    assert "pageSize=5" in str(request.url)


@respx.mock
async def test_async_list_files_page_with_pagination_token(
    async_assistants: AsyncAssistants,
) -> None:
    """list_files_page() sends paginationToken query param when provided."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    await async_assistants.list_files_page(
        assistant_name="test-assistant", pagination_token="tok123"
    )

    request = route.calls.last.request
    assert "paginationToken=tok123" in str(request.url)


@respx.mock
async def test_async_list_files_page_with_filter(
    async_assistants: AsyncAssistants,
) -> None:
    """list_files_page() serializes filter dict to JSON string query param."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    await async_assistants.list_files_page(
        assistant_name="test-assistant",
        filter={"genre": {"$eq": "comedy"}},
    )

    request = route.calls.last.request
    url_str = str(request.url)
    assert "filter=" in url_str
    assert "genre" in url_str


@respx.mock
async def test_async_list_files_page_omits_none_params(
    async_assistants: AsyncAssistants,
) -> None:
    """list_files_page() does not send params that are None."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    await async_assistants.list_files_page(assistant_name="test-assistant")

    request = route.calls.last.request
    assert "pageSize" not in str(request.url)
    assert "paginationToken" not in str(request.url)
    assert "filter" not in str(request.url)


# ---------------------------------------------------------------------------
# list_files() — auto-pagination
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_list_files_empty(async_assistants: AsyncAssistants) -> None:
    """list_files() returns an empty list when no files exist."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    result = await async_assistants.list_files(assistant_name="test-assistant")

    assert result == []


@respx.mock
async def test_async_list_files_single_page(async_assistants: AsyncAssistants) -> None:
    """list_files() returns all files from a single-page response."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={
                "files": [
                    make_assistant_file_response(id="f1", name="file1.pdf"),
                    make_assistant_file_response(id="f2", name="file2.pdf"),
                ],
            },
        ),
    )

    result = await async_assistants.list_files(assistant_name="test-assistant")

    assert len(result) == 2
    assert all(isinstance(f, AssistantFileModel) for f in result)
    assert result[0].id == "f1"
    assert result[1].id == "f2"


@respx.mock
async def test_async_list_files_multi_page(async_assistants: AsyncAssistants) -> None:
    """list_files() auto-paginates through multiple pages collecting all files."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "files": [make_assistant_file_response(id="f1", name="file1.pdf")],
                    "next": "token-page2",
                },
            ),
            httpx.Response(
                200,
                json={
                    "files": [make_assistant_file_response(id="f2", name="file2.pdf")],
                    "next": "token-page3",
                },
            ),
            httpx.Response(
                200,
                json={
                    "files": [make_assistant_file_response(id="f3", name="file3.pdf")],
                },
            ),
        ]
    )

    result = await async_assistants.list_files(assistant_name="test-assistant")

    assert len(result) == 3
    assert [f.id for f in result] == ["f1", "f2", "f3"]


@respx.mock
async def test_async_list_files_with_filter(async_assistants: AsyncAssistants) -> None:
    """list_files() passes filter to list_files_page on each request."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(host=DATA_PLANE_HOST)),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={"files": [make_assistant_file_response()]},
        ),
    )

    result = await async_assistants.list_files(
        assistant_name="test-assistant",
        filter={"genre": {"$eq": "comedy"}},
    )

    assert len(result) == 1
    request = route.calls.last.request
    url_str = str(request.url)
    assert "filter=" in url_str
    assert "genre" in url_str
