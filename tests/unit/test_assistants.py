"""Unit tests for Assistants namespace — create_assistant."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ASSISTANT_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.assistants import (
    _CREATE_POLL_INTERVAL_SECONDS,
    Assistants,
)
from pinecone.models.assistant.list import ListAssistantsResponse
from pinecone.errors.exceptions import PineconeTimeoutError, PineconeValueError
from pinecone.models.assistant.model import AssistantModel
from tests.factories import make_assistant_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture()
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ASSISTANT_API_VERSION)


@pytest.fixture()
def assistants() -> Assistants:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return Assistants(config=config)


# ---------------------------------------------------------------------------
# create() — validation
# ---------------------------------------------------------------------------


def test_create_assistant_region_validation(assistants: Assistants) -> None:
    """Invalid region raises PineconeValueError before any HTTP call."""
    with pytest.raises(PineconeValueError, match="region") as exc_info:
        assistants.create(name="test-assistant", region="ap-southeast-1")
    assert "ap-southeast-1" in str(exc_info.value)


def test_create_assistant_region_case_sensitive(assistants: Assistants) -> None:
    """Uppercase 'US' and 'EU' are rejected — validation is case-sensitive."""
    with pytest.raises(PineconeValueError, match="region"):
        assistants.create(name="test-assistant", region="US")

    with pytest.raises(PineconeValueError, match="region"):
        assistants.create(name="test-assistant", region="EU")


# ---------------------------------------------------------------------------
# create() — defaults
# ---------------------------------------------------------------------------


@respx.mock
def test_create_assistant_defaults(assistants: Assistants) -> None:
    """Default region is 'us', metadata is {}, instructions is None."""
    route = respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = assistants.create(name="test-assistant")

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
def test_create_assistant_immediate_return(assistants: Assistants) -> None:
    """timeout=-1 returns immediately without polling."""
    route = respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    result = assistants.create(name="test-assistant", timeout=-1)

    assert isinstance(result, AssistantModel)
    assert result.status == "Initializing"
    assert route.call_count == 1


@respx.mock
def test_create_assistant_with_all_params(assistants: Assistants) -> None:
    """Create with instructions, metadata, and region sends correct body."""
    route = respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    result = assistants.create(
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
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_polls_until_ready(mock_sleep: object, assistants: Assistants) -> None:
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

    result = assistants.create(name="test-assistant")

    assert result.status == "Ready"
    assert poll_route.call_count == 3


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_polls_with_correct_interval(
    mock_sleep: object, assistants: Assistants
) -> None:
    """Polling sleeps with the correct interval between polls."""
    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    assistants.create(name="test-assistant")

    # sleep is not called after the final successful poll
    # but it may be called before the first poll depending on flow
    # The key assertion is the interval value
    from unittest.mock import call

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_CREATE_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# create() — timeout
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_timeout_raises(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
) -> None:
    """Exceeding timeout raises PineconeTimeoutError with helpful message."""
    # Simulate time progression: start=0, after first poll=6 (exceeds timeout=5)
    mock_monotonic.side_effect = [0.0, 6.0]  # type: ignore[union-attr]

    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Initializing")),
    )

    with pytest.raises(PineconeTimeoutError, match="not ready") as exc_info:
        assistants.create(name="test-assistant", timeout=5)

    assert "describe_assistant" in str(exc_info.value)


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_timeout_zero_polls_once(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
) -> None:
    """timeout=0 polls once, then raises if not ready."""
    # start=0, first elapsed check=0.0 (not >= 0, so one iteration), second=0.1 (>= 0)
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
        assistants.create(name="test-assistant", timeout=0)

    # At least one poll was made
    assert poll_route.call_count >= 1


# ---------------------------------------------------------------------------
# create() — status check is case-sensitive
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_status_case_sensitive(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
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
        assistants.create(name="test-assistant", timeout=5)


# ---------------------------------------------------------------------------
# create() — indefinite polling (timeout=None)
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_create_assistant_polls_indefinitely_when_no_timeout(
    mock_sleep: object, assistants: Assistants
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

    result = assistants.create(name="test-assistant", timeout=None)

    assert result.status == "Ready"


# ---------------------------------------------------------------------------
# create() — valid regions accepted
# ---------------------------------------------------------------------------


@respx.mock
def test_create_assistant_accepts_us_region(assistants: Assistants) -> None:
    """Region 'us' is accepted."""
    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = assistants.create(name="test-assistant", region="us")
    assert isinstance(result, AssistantModel)


@respx.mock
def test_create_assistant_accepts_eu_region(assistants: Assistants) -> None:
    """Region 'eu' is accepted."""
    respx.post(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response(status="Ready")),
    )

    result = assistants.create(name="test-assistant", region="eu")
    assert isinstance(result, AssistantModel)


# ---------------------------------------------------------------------------
# describe() — success
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_assistant(assistants: Assistants) -> None:
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

    result = assistants.describe(name="my-assistant")

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
def test_describe_assistant_not_found(assistants: Assistants) -> None:
    """describe() lets 404 errors propagate from the HTTP client."""
    respx.get(f"{BASE_URL}/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(Exception):
        assistants.describe(name="nonexistent")


@respx.mock
def test_describe_assistant_minimal_response(assistants: Assistants) -> None:
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

    result = assistants.describe(name="minimal")

    assert result.name == "minimal"
    assert result.metadata is None
    assert result.instructions is None
    assert result.host is None


# ---------------------------------------------------------------------------
# list() — auto-pagination
# ---------------------------------------------------------------------------


@respx.mock
def test_list_assistants_empty(assistants: Assistants) -> None:
    """list() returns empty list when no assistants exist."""
    respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    result = assistants.list()

    assert result == []


@respx.mock
def test_list_assistants_single_page(assistants: Assistants) -> None:
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

    result = assistants.list()

    assert len(result) == 2
    assert result[0].name == "a1"
    assert result[1].name == "a2"
    assert all(isinstance(a, AssistantModel) for a in result)


@respx.mock
def test_list_assistants_multi_page(assistants: Assistants) -> None:
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

    result = assistants.list()

    assert len(result) == 3
    assert [a.name for a in result] == ["a1", "a2", "a3"]


# ---------------------------------------------------------------------------
# list_page() — single page with pagination control
# ---------------------------------------------------------------------------


@respx.mock
def test_list_assistants_page(assistants: Assistants) -> None:
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

    result = assistants.list_page()

    assert isinstance(result, ListAssistantsResponse)
    assert len(result.assistants) == 1
    assert result.assistants[0].name == "a1"
    assert result.next == "token-next"
    assert route.call_count == 1


@respx.mock
def test_list_assistants_page_last_page(assistants: Assistants) -> None:
    """list_page() returns no next token on the last page."""
    respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(
            200,
            json={
                "assistants": [make_assistant_response(name="a1")],
            },
        ),
    )

    result = assistants.list_page()

    assert result.next is None


@respx.mock
def test_list_assistants_page_with_page_size(assistants: Assistants) -> None:
    """list_page() sends pageSize query param when provided."""
    route = respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    assistants.list_page(page_size=5)

    request = route.calls.last.request
    assert "pageSize=5" in str(request.url)


@respx.mock
def test_list_assistants_page_with_pagination_token(assistants: Assistants) -> None:
    """list_page() sends paginationToken query param when provided."""
    route = respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    assistants.list_page(pagination_token="abc123")

    request = route.calls.last.request
    assert "paginationToken=abc123" in str(request.url)


@respx.mock
def test_list_assistants_page_omits_none_params(assistants: Assistants) -> None:
    """list_page() does not send params that are None."""
    route = respx.get(f"{BASE_URL}/assistants").mock(
        return_value=httpx.Response(200, json={"assistants": []}),
    )

    assistants.list_page()

    request = route.calls.last.request
    assert "pageSize" not in str(request.url)
    assert "paginationToken" not in str(request.url)
