"""Unit tests for Assistants namespace — create_assistant."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone import Pinecone
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ASSISTANT_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.assistants import (
    _CREATE_POLL_INTERVAL_SECONDS,
    _DELETE_POLL_INTERVAL_SECONDS,
    _UPLOAD_POLL_INTERVAL_SECONDS,
    Assistants,
)
from pinecone.errors.exceptions import (
    NotFoundError,
    PineconeError,
    PineconeTimeoutError,
    PineconeValueError,
)
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListAssistantsResponse
from pinecone.models.assistant.model import AssistantModel
from tests.factories import make_assistant_file_response, make_assistant_response

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

    assert "pc.assistants.describe" in str(exc_info.value)


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

    with pytest.raises(NotFoundError):
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


# ---------------------------------------------------------------------------
# update() — success
# ---------------------------------------------------------------------------


@respx.mock
def test_update_assistant_instructions(assistants: Assistants) -> None:
    """update() sends PATCH /assistants/{name} with instructions and returns AssistantModel."""
    updated_response = make_assistant_response(
        name="my-assistant",
        instructions="Updated instructions.",
    )
    route = respx.patch(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = assistants.update(name="my-assistant", instructions="Updated instructions.")

    assert isinstance(result, AssistantModel)
    assert result.name == "my-assistant"
    assert result.instructions == "Updated instructions."
    assert route.call_count == 1

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"instructions": "Updated instructions."}


@respx.mock
def test_update_assistant_metadata(assistants: Assistants) -> None:
    """update() sends PATCH with metadata and returns AssistantModel."""
    new_metadata = {"team": "ml", "version": "2"}
    updated_response = make_assistant_response(
        name="my-assistant",
        metadata=new_metadata,
    )
    route = respx.patch(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = assistants.update(name="my-assistant", metadata=new_metadata)

    assert isinstance(result, AssistantModel)
    assert result.metadata == new_metadata
    assert route.call_count == 1

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"metadata": new_metadata}


@respx.mock
def test_update_assistant_both_fields(assistants: Assistants) -> None:
    """update() sends both instructions and metadata when provided."""
    updated_response = make_assistant_response(
        name="my-assistant",
        instructions="New instructions.",
        metadata={"env": "prod"},
    )
    route = respx.patch(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = assistants.update(
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
def test_update_assistant_clear_instructions(assistants: Assistants) -> None:
    """update() can clear instructions by setting them to empty string."""
    updated_response = make_assistant_response(
        name="my-assistant",
        instructions="",
    )
    route = respx.patch(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    result = assistants.update(name="my-assistant", instructions="")

    assert result.instructions == ""

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {"instructions": ""}


@respx.mock
def test_update_assistant_omits_none_fields(assistants: Assistants) -> None:
    """update() only includes provided fields in the request body."""
    updated_response = make_assistant_response(name="my-assistant")
    route = respx.patch(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=updated_response),
    )

    assistants.update(name="my-assistant", instructions="Only this.")

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "metadata" not in body
    assert body == {"instructions": "Only this."}


@respx.mock
def test_update_assistant_not_found(assistants: Assistants) -> None:
    """update() lets 404 errors propagate from the HTTP client."""
    respx.patch(f"{BASE_URL}/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        assistants.update(name="nonexistent", instructions="test")


# ---------------------------------------------------------------------------
# list_page() — omission
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# delete() — polls until gone
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_assistant(mock_sleep: object, assistants: Assistants) -> None:
    """delete() sends DELETE then polls describe until 404 confirms deletion."""
    delete_route = respx.delete(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    result = assistants.delete(name="my-assistant")

    assert result is None
    assert delete_route.call_count == 1

    request = delete_route.calls.last.request
    assert request.method == "DELETE"
    assert str(request.url) == f"{BASE_URL}/assistants/my-assistant"


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_assistant_polls_until_gone(mock_sleep: object, assistants: Assistants) -> None:
    """delete() polls describe every 5s; returns when 404 is received."""
    respx.delete(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    describe_route = respx.get(f"{BASE_URL}/assistants/my-assistant").mock(
        side_effect=[
            httpx.Response(
                200, json=make_assistant_response(name="my-assistant", status="Terminating")
            ),
            httpx.Response(
                200, json=make_assistant_response(name="my-assistant", status="Terminating")
            ),
            httpx.Response(404, json={"error": "Not found"}),
        ],
    )

    result = assistants.delete(name="my-assistant")

    assert result is None
    assert describe_route.call_count == 3

    from unittest.mock import call

    from pinecone.client.assistants import _DELETE_POLL_INTERVAL_SECONDS

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_DELETE_POLL_INTERVAL_SECONDS)


@respx.mock
def test_delete_assistant_timeout_minus_one_skips_polling(assistants: Assistants) -> None:
    """delete(timeout=-1) returns immediately without polling."""
    delete_route = respx.delete(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )

    result = assistants.delete(name="my-assistant", timeout=-1)

    assert result is None
    assert delete_route.call_count == 1


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_delete_assistant_timeout_raises(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
) -> None:
    """Exceeding timeout raises PineconeTimeoutError."""
    mock_monotonic.side_effect = [0.0, 11.0]  # type: ignore[union-attr]

    respx.delete(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(
            200, json=make_assistant_response(name="my-assistant", status="Terminating")
        ),
    )

    with pytest.raises(PineconeTimeoutError, match="still exists after 10"):
        assistants.delete(name="my-assistant", timeout=10)


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_assistant_polls_indefinitely_when_no_timeout(
    mock_sleep: object, assistants: Assistants
) -> None:
    """When timeout is None, polling continues until 404 with no deadline."""
    respx.delete(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistants/my-assistant").mock(
        side_effect=[
            httpx.Response(
                200, json=make_assistant_response(name="my-assistant", status="Terminating")
            ),
            httpx.Response(
                200, json=make_assistant_response(name="my-assistant", status="Terminating")
            ),
            httpx.Response(
                200, json=make_assistant_response(name="my-assistant", status="Terminating")
            ),
            httpx.Response(404, json={"error": "Not found"}),
        ],
    )

    result = assistants.delete(name="my-assistant")

    assert result is None


@respx.mock
def test_delete_assistant_not_found(assistants: Assistants) -> None:
    """delete() lets 404 errors propagate from the initial DELETE request."""
    respx.delete(f"{BASE_URL}/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        assistants.delete(name="nonexistent")


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_assistant_propagates_non_404_errors_during_poll(
    mock_sleep: object, assistants: Assistants
) -> None:
    """Non-404 errors during polling propagate instead of being swallowed."""
    from pinecone.errors.exceptions import ServiceError

    respx.delete(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(204),
    )
    respx.get(f"{BASE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(500, json={"error": "Internal server error"}),
    )

    with pytest.raises(ServiceError):
        assistants.delete(name="my-assistant")


# ---------------------------------------------------------------------------
# list_page() — omission
# ---------------------------------------------------------------------------


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


CONTROL_PLANE_URL = "https://api.pinecone.io"


# ---------------------------------------------------------------------------
# Pinecone.assistant() convenience method
# ---------------------------------------------------------------------------


@respx.mock
def test_pinecone_assistant_convenience_method() -> None:
    """pc.assistant(name=...) delegates to GET /assistants/{name} and returns AssistantModel."""
    response_data = make_assistant_response(
        name="my-assistant",
        status="Ready",
        instructions="Be helpful.",
        metadata={"team": "ml"},
        host="my-assistant-abc.svc.pinecone.io",
    )
    route = respx.get(f"{CONTROL_PLANE_URL}/assistants/my-assistant").mock(
        return_value=httpx.Response(200, json=response_data),
    )

    pc = Pinecone(api_key="test-key")
    result = pc.assistant(name="my-assistant")

    assert isinstance(result, AssistantModel)
    assert result.name == "my-assistant"
    assert result.status == "Ready"
    assert result.instructions == "Be helpful."
    assert result.metadata == {"team": "ml"}
    assert result.host == "my-assistant-abc.svc.pinecone.io"
    assert route.call_count == 1


@respx.mock
def test_pinecone_assistant_not_found() -> None:
    """pc.assistant(name=...) raises NotFoundError when assistant does not exist."""
    respx.get(f"{CONTROL_PLANE_URL}/assistants/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    pc = Pinecone(api_key="test-key")
    with pytest.raises(NotFoundError):
        pc.assistant(name="nonexistent")


# ---------------------------------------------------------------------------
# upload_file() — validation
# ---------------------------------------------------------------------------

DATA_PLANE_HOST = "test-assistant-abc123.svc.pinecone.io"
DATA_PLANE_URL = f"https://{DATA_PLANE_HOST}"


def test_upload_file_path_not_found(assistants: Assistants) -> None:
    """Uploading from a nonexistent local path raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="File not found"):
        assistants.upload_file(
            assistant_name="test-assistant",
            file_path="/nonexistent/path/document.pdf",
        )


def test_upload_file_neither_path_nor_stream(assistants: Assistants) -> None:
    """Providing neither file_path nor file_stream raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="Exactly one"):
        assistants.upload_file(assistant_name="test-assistant")


def test_upload_file_both_path_and_stream(assistants: Assistants) -> None:
    """Providing both file_path and file_stream raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="Exactly one"):
        assistants.upload_file(
            assistant_name="test-assistant",
            file_path="/some/path.pdf",
            file_stream=io.BytesIO(b"data"),
        )


# ---------------------------------------------------------------------------
# upload_file() — multimodal serialization
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_multimodal_serialization(mock_sleep: object, assistants: Assistants) -> None:
    """multimodal bool is converted to lowercase string in query params."""
    # Mock describe call for data-plane host resolution
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )

    # Mock the upload POST on data-plane
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )

    # Mock the poll GET on data-plane
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    stream = io.BytesIO(b"file content")
    assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        file_name="doc.pdf",
        multimodal=True,
    )

    request = upload_route.calls.last.request
    assert "multimodal=true" in str(request.url)


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_multimodal_false_serialization(
    mock_sleep: object, assistants: Assistants
) -> None:
    """multimodal=False is sent as 'false' string."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    stream = io.BytesIO(b"file content")
    assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        file_name="doc.pdf",
        multimodal=False,
    )

    request = upload_route.calls.last.request
    assert "multimodal=false" in str(request.url)


# ---------------------------------------------------------------------------
# upload_file() — success from stream
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_from_stream(mock_sleep: object, assistants: Assistants) -> None:
    """upload_file with file_stream uploads and polls until Available."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    poll_route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_file_response(status="Processing")),
            httpx.Response(200, json=make_assistant_file_response(status="Available")),
        ]
    )

    stream = io.BytesIO(b"test file content")
    result = assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        file_name="report.pdf",
    )

    assert isinstance(result, AssistantFileModel)
    assert result.status == "Available"
    assert result.id == "file-abc123"
    assert upload_route.call_count == 1
    assert poll_route.call_count == 2


# ---------------------------------------------------------------------------
# upload_file() — success with metadata and file_id
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_with_metadata_and_file_id(mock_sleep: object, assistants: Assistants) -> None:
    """upload_file sends metadata as JSON string and file_id as query params."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    upload_route = respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    stream = io.BytesIO(b"content")
    assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
        metadata={"genre": "comedy"},
        file_id="custom-file-id",
    )

    request = upload_route.calls.last.request
    url_str = str(request.url)
    assert "file_id=custom-file-id" in url_str
    # Metadata is JSON-encoded
    assert "metadata=" in url_str


# ---------------------------------------------------------------------------
# upload_file() — polling
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_polls_with_correct_interval(
    mock_sleep: object, assistants: Assistants
) -> None:
    """upload_file polls every 5 seconds while processing."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_file_response(status="Processing")),
            httpx.Response(200, json=make_assistant_file_response(status="Processing")),
            httpx.Response(200, json=make_assistant_file_response(status="Available")),
        ]
    )

    stream = io.BytesIO(b"data")
    assistants.upload_file(
        assistant_name="test-assistant",
        file_stream=stream,
    )

    from unittest.mock import call

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_UPLOAD_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# upload_file() — processing failure
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_processing_failed(mock_sleep: object, assistants: Assistants) -> None:
    """If processing fails, raises PineconeError with server's error message."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_file_response(
                status="ProcessingFailed",
                error_message="Unsupported file format",
            ),
        ),
    )

    stream = io.BytesIO(b"data")
    with pytest.raises(PineconeError, match="Unsupported file format"):
        assistants.upload_file(
            assistant_name="test-assistant",
            file_stream=stream,
        )


# ---------------------------------------------------------------------------
# upload_file() — timeout
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_timeout_raises(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
) -> None:
    """Timeout raises PineconeTimeoutError with operation ID in message."""
    mock_monotonic.side_effect = [0.0, 11.0]  # type: ignore[union-attr]

    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Processing")),
    )

    stream = io.BytesIO(b"data")
    with pytest.raises(PineconeTimeoutError, match="file-abc123") as exc_info:
        assistants.upload_file(
            assistant_name="test-assistant",
            file_stream=stream,
            timeout=10,
        )

    assert "operation_id" in str(exc_info.value)


# ---------------------------------------------------------------------------
# upload_file() — data-plane client caching
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_upload_file_caches_data_plane_client(mock_sleep: object, assistants: Assistants) -> None:
    """Second upload to same assistant reuses cached data-plane client."""
    describe_route = respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.post(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Available")),
    )

    # First upload — triggers describe
    stream1 = io.BytesIO(b"data1")
    assistants.upload_file(assistant_name="test-assistant", file_stream=stream1)

    # Second upload — should reuse cached client
    stream2 = io.BytesIO(b"data2")
    assistants.upload_file(assistant_name="test-assistant", file_stream=stream2)

    # Describe should only be called once (for the first upload)
    assert describe_route.call_count == 1


# ---------------------------------------------------------------------------
# describe_file() — success
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_file_success(assistants: Assistants) -> None:
    """describe_file() sends GET /files/{name}/{id} and returns AssistantFileModel."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response()),
    )

    result = assistants.describe_file(assistant_name="test-assistant", file_id="file-abc123")

    assert isinstance(result, AssistantFileModel)
    assert result.id == "file-abc123"
    assert result.name == "test-file.pdf"
    assert route.call_count == 1


@respx.mock
def test_describe_file_without_url(assistants: Assistants) -> None:
    """describe_file() does not send include_url param by default."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response()),
    )

    assistants.describe_file(assistant_name="test-assistant", file_id="file-abc123")

    request = route.calls.last.request
    assert "include_url" not in str(request.url)


@respx.mock
def test_describe_file_with_url(assistants: Assistants) -> None:
    """describe_file(include_url=True) sends include_url=true query param."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_file_response(
                signed_url="https://storage.example.com/file-abc123"
            ),
        ),
    )

    result = assistants.describe_file(
        assistant_name="test-assistant", file_id="file-abc123", include_url=True
    )

    request = route.calls.last.request
    assert "include_url=true" in str(request.url)
    assert result.signed_url == "https://storage.example.com/file-abc123"


@respx.mock
def test_describe_file_not_found(assistants: Assistants) -> None:
    """describe_file() raises NotFoundError when file does not exist."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    with pytest.raises(NotFoundError):
        assistants.describe_file(assistant_name="test-assistant", file_id="nonexistent")


# ---------------------------------------------------------------------------
# list_files_page() — single page
# ---------------------------------------------------------------------------


@respx.mock
def test_list_files_page_success(assistants: Assistants) -> None:
    """list_files_page() returns ListFilesResponse with files and next token."""
    from pinecone.models.assistant.list import ListFilesResponse

    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
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

    result = assistants.list_files_page(assistant_name="test-assistant")

    assert isinstance(result, ListFilesResponse)
    assert len(result.files) == 1
    assert result.files[0].id == "file-abc123"
    assert result.next == "token-next"
    assert route.call_count == 1


@respx.mock
def test_list_files_page_last_page(assistants: Assistants) -> None:
    """list_files_page() returns no next token on the last page."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": [make_assistant_file_response()]}),
    )

    result = assistants.list_files_page(assistant_name="test-assistant")

    assert result.next is None


@respx.mock
def test_list_files_page_with_page_size(assistants: Assistants) -> None:
    """list_files_page() sends pageSize query param when provided."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    assistants.list_files_page(assistant_name="test-assistant", page_size=5)

    request = route.calls.last.request
    assert "pageSize=5" in str(request.url)


@respx.mock
def test_list_files_page_with_pagination_token(assistants: Assistants) -> None:
    """list_files_page() sends paginationToken query param when provided."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    assistants.list_files_page(assistant_name="test-assistant", pagination_token="tok123")

    request = route.calls.last.request
    assert "paginationToken=tok123" in str(request.url)


@respx.mock
def test_list_files_page_with_filter(assistants: Assistants) -> None:
    """list_files_page() serializes filter dict to JSON string query param."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    assistants.list_files_page(
        assistant_name="test-assistant",
        filter={"genre": {"$eq": "comedy"}},
    )

    request = route.calls.last.request
    url_str = str(request.url)
    assert "filter=" in url_str
    assert "genre" in url_str


@respx.mock
def test_list_files_page_omits_none_params(assistants: Assistants) -> None:
    """list_files_page() does not send params that are None."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    assistants.list_files_page(assistant_name="test-assistant")

    request = route.calls.last.request
    assert "pageSize" not in str(request.url)
    assert "paginationToken" not in str(request.url)
    assert "filter" not in str(request.url)


# ---------------------------------------------------------------------------
# list_files() — auto-pagination
# ---------------------------------------------------------------------------


@respx.mock
def test_list_files_empty(assistants: Assistants) -> None:
    """list_files() returns empty list when no files exist."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        return_value=httpx.Response(200, json={"files": []}),
    )

    result = assistants.list_files(assistant_name="test-assistant")

    assert result == []


@respx.mock
def test_list_files_single_page(assistants: Assistants) -> None:
    """list_files() returns all files from a single-page response."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
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

    result = assistants.list_files(assistant_name="test-assistant")

    assert len(result) == 2
    assert all(isinstance(f, AssistantFileModel) for f in result)


@respx.mock
def test_list_files_multi_page(assistants: Assistants) -> None:
    """list_files() auto-paginates through multiple pages."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
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

    result = assistants.list_files(assistant_name="test-assistant")

    assert len(result) == 3
    assert [f.id for f in result] == ["f1", "f2", "f3"]


@respx.mock
def test_list_files_propagates_filter_through_pages(assistants: Assistants) -> None:
    """list_files() sends the same filter on every paginated request."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "files": [make_assistant_file_response(id="f1", name="file1.pdf")],
                    "next": "token-p2",
                },
            ),
            httpx.Response(
                200,
                json={"files": [make_assistant_file_response(id="f2", name="file2.pdf")]},
            ),
        ]
    )

    assistants.list_files(
        assistant_name="test-assistant",
        filter={"genre": {"$eq": "comedy"}},
    )

    assert route.call_count == 2
    for call_obj in route.calls:
        assert "filter=" in str(call_obj.request.url)
        assert "genre" in str(call_obj.request.url)


# ---------------------------------------------------------------------------
# delete_file() — success
# ---------------------------------------------------------------------------


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_file_success(mock_sleep: object, assistants: Assistants) -> None:
    """delete_file() sends DELETE then polls until 404 confirms deletion."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    delete_route = respx.delete(
        f"{DATA_PLANE_URL}/files/test-assistant/file-abc123"
    ).mock(return_value=httpx.Response(200))
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(404, json={"error": "Not found"}),
    )

    result = assistants.delete_file(assistant_name="test-assistant", file_id="file-abc123")

    assert result is None
    assert delete_route.call_count == 1


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_file_polls_until_gone(mock_sleep: object, assistants: Assistants) -> None:
    """delete_file() polls describe_file every 5s; returns when 404 received."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200)
    )
    poll_route = respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        side_effect=[
            httpx.Response(200, json=make_assistant_file_response(status="Deleting")),
            httpx.Response(200, json=make_assistant_file_response(status="Deleting")),
            httpx.Response(404, json={"error": "Not found"}),
        ]
    )

    result = assistants.delete_file(assistant_name="test-assistant", file_id="file-abc123")

    assert result is None
    assert poll_route.call_count == 3

    from unittest.mock import call

    for c in mock_sleep.call_args_list:  # type: ignore[union-attr]
        assert c == call(_DELETE_POLL_INTERVAL_SECONDS)


@respx.mock
def test_delete_file_timeout_minus_one_skips_polling(assistants: Assistants) -> None:
    """delete_file(timeout=-1) returns immediately without polling."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    delete_route = respx.delete(
        f"{DATA_PLANE_URL}/files/test-assistant/file-abc123"
    ).mock(return_value=httpx.Response(200))

    result = assistants.delete_file(
        assistant_name="test-assistant", file_id="file-abc123", timeout=-1
    )

    assert result is None
    assert delete_route.call_count == 1


@respx.mock
@patch("pinecone.client.assistants.time.monotonic")
@patch("pinecone.client.assistants.time.sleep")
def test_delete_file_timeout_raises(
    mock_sleep: object, mock_monotonic: object, assistants: Assistants
) -> None:
    """Exceeding timeout raises PineconeTimeoutError."""
    mock_monotonic.side_effect = [0.0, 11.0]  # type: ignore[union-attr]

    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200)
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200, json=make_assistant_file_response(status="Deleting")),
    )

    with pytest.raises(PineconeTimeoutError, match="still exists after 10"):
        assistants.delete_file(
            assistant_name="test-assistant", file_id="file-abc123", timeout=10
        )


@respx.mock
@patch("pinecone.client.assistants.time.sleep")
def test_delete_file_server_error_raises(mock_sleep: object, assistants: Assistants) -> None:
    """delete_file() raises PineconeError if server-side deletion fails."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    respx.delete(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(200)
    )
    respx.get(f"{DATA_PLANE_URL}/files/test-assistant/file-abc123").mock(
        return_value=httpx.Response(
            200,
            json=make_assistant_file_response(
                status="ProcessingFailed", error_message="Storage backend error"
            ),
        ),
    )

    with pytest.raises(PineconeError, match="Storage backend error"):
        assistants.delete_file(assistant_name="test-assistant", file_id="file-abc123")


# ---------------------------------------------------------------------------
# chat() — validation
# ---------------------------------------------------------------------------


def test_chat_json_streaming_validation(assistants: Assistants) -> None:
    """Requesting json_response=True with stream=True raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="json_response"):
        assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
            json_response=True,
        )


# ---------------------------------------------------------------------------
# chat() — defaults
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_default_model(assistants: Assistants) -> None:
    """chat() defaults model to 'gpt-4o' when not specified."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chat-abc123",
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
                "citations": [],
            },
        )
    )

    from pinecone.models.assistant.chat import ChatResponse

    result = assistants.chat(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
    )

    assert isinstance(result, ChatResponse)
    request_body = json.loads(chat_route.calls.last.request.content)
    assert request_body["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# chat() — message parsing
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_message_parsing(assistants: Assistants) -> None:
    """Dicts are converted to Message objects; missing role defaults to 'user'."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chat-abc123",
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop",
                "citations": [],
            },
        )
    )

    assistants.chat(
        assistant_name="test-assistant",
        messages=[
            {"content": "No role here"},
            {"content": "Explicit role", "role": "user"},
        ],
    )

    request_body = json.loads(chat_route.calls.last.request.content)
    msgs = request_body["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "No role here"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Explicit role"


# ---------------------------------------------------------------------------
# chat_completions() — defaults
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_completions_default_stream_false(assistants: Assistants) -> None:
    """chat_completions() defaults stream to False and posts to the completions endpoint."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    completions_route = respx.post(
        f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-abc123",
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from completions!"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    from pinecone.models.assistant.chat import ChatCompletionResponse

    result = assistants.chat_completions(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
    )

    assert isinstance(result, ChatCompletionResponse)
    request_body = json.loads(completions_route.calls.last.request.content)
    assert request_body["stream"] is False
    assert request_body["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# chat_completions() — no model validation
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_completions_no_model_validation(assistants: Assistants) -> None:
    """chat_completions() accepts any model string without client-side validation."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    completions_route = respx.post(
        f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-xyz789",
                "model": "some-unknown-model-v99",
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )

    from pinecone.models.assistant.chat import ChatCompletionResponse

    result = assistants.chat_completions(
        assistant_name="test-assistant",
        messages=[{"content": "Hello"}],
        model="some-unknown-model-v99",
    )

    assert isinstance(result, ChatCompletionResponse)
    request_body = json.loads(completions_route.calls.last.request.content)
    assert request_body["model"] == "some-unknown-model-v99"


# ---------------------------------------------------------------------------
# _chat_streaming — SSE parsing
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_streaming_sse_parsing(assistants: Assistants) -> None:
    """Empty SSE lines are skipped and 'data:' prefix is stripped before JSON parsing."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    # SSE payload: empty lines between events, data: prefix on each event
    sse_body = (
        b"data: {\"type\": \"message_start\", \"model\": \"gpt-4o\", \"role\": \"assistant\"}\n"
        b"\n"
        b"data: {\"type\": \"message_end\", \"id\": \"end1\","
        b" \"usage\": {\"prompt_tokens\": 5, \"completion_tokens\": 10, \"total_tokens\": 15}}\n"
        b"\n"
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    from pinecone.models.assistant.streaming import StreamMessageEnd, StreamMessageStart

    chunks = list(
        assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    assert len(chunks) == 2
    assert isinstance(chunks[0], StreamMessageStart)
    assert isinstance(chunks[1], StreamMessageEnd)


# ---------------------------------------------------------------------------
# _chat_streaming — chunk dispatch
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_streaming_chunk_dispatch(assistants: Assistants) -> None:
    """Correct chunk types are yielded based on the 'type' field in each SSE event."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b"data: {\"type\": \"message_start\", \"model\": \"gpt-4o\", \"role\": \"assistant\"}\n"
        b"data: {\"type\": \"content_chunk\", \"id\": \"c1\","
        b" \"delta\": {\"content\": \"Hello\"}}\n"
        b"data: {\"type\": \"citation\", \"id\": \"cit1\","
        b" \"citation\": {\"position\": 5, \"references\": []}}\n"
        b"data: {\"type\": \"message_end\", \"id\": \"end1\","
        b" \"usage\": {\"prompt_tokens\": 5, \"completion_tokens\": 10, \"total_tokens\": 15}}\n"
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    from pinecone.models.assistant.streaming import (
        StreamCitationChunk,
        StreamContentChunk,
        StreamMessageEnd,
        StreamMessageStart,
    )

    chunks = list(
        assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    assert len(chunks) == 4
    assert isinstance(chunks[0], StreamMessageStart)
    assert chunks[0].model == "gpt-4o"
    assert chunks[0].role == "assistant"
    assert isinstance(chunks[1], StreamContentChunk)
    assert chunks[1].delta.content == "Hello"
    assert isinstance(chunks[2], StreamCitationChunk)
    assert isinstance(chunks[3], StreamMessageEnd)
    assert chunks[3].usage.total_tokens == 15


# ---------------------------------------------------------------------------
# _chat_streaming — request shape
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_streaming_request_body(assistants: Assistants) -> None:
    """Streaming chat always includes stream=True and include_highlights in body."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b"data: {\"type\": \"message_start\", \"model\": \"gpt-4o\", \"role\": \"assistant\"}\n"
        b"data: {\"type\": \"message_end\", \"id\": \"e1\","
        b" \"usage\": {\"prompt_tokens\": 1, \"completion_tokens\": 1, \"total_tokens\": 2}}\n"
    )
    chat_route = respx.post(f"{DATA_PLANE_URL}/chat/test-assistant").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    list(
        assistants.chat(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    request_body = json.loads(chat_route.calls.last.request.content)
    assert request_body["stream"] is True
    assert "include_highlights" in request_body
    assert request_body["include_highlights"] is False


# ---------------------------------------------------------------------------
# _chat_completions_streaming — SSE parsing
# ---------------------------------------------------------------------------


@respx.mock
def test_chat_completions_streaming_sse_parsing(assistants: Assistants) -> None:
    """chat_completions streaming parses SSE lines as ChatCompletionStreamChunk."""
    respx.get(f"{BASE_URL}/assistants/test-assistant").mock(
        return_value=httpx.Response(200, json=make_assistant_response()),
    )
    sse_body = (
        b"data: {\"id\": \"cmpl1\", \"choices\": [{\"index\": 0,"
        b" \"delta\": {\"role\": \"assistant\", \"content\": null},"
        b" \"finish_reason\": null}]}\n"
        b"\n"
        b"data: {\"id\": \"cmpl2\", \"choices\": [{\"index\": 0,"
        b" \"delta\": {\"content\": \"Hello\"},"
        b" \"finish_reason\": null}]}\n"
        b"\n"
        b"data: {\"id\": \"cmpl3\", \"choices\": [{\"index\": 0,"
        b" \"delta\": {}, \"finish_reason\": \"stop\"}]}\n"
    )
    respx.post(f"{DATA_PLANE_URL}/chat/test-assistant/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_body),
    )

    from pinecone.models.assistant.streaming import ChatCompletionStreamChunk

    chunks = list(
        assistants.chat_completions(
            assistant_name="test-assistant",
            messages=[{"content": "Hello"}],
            stream=True,
        )
    )

    assert len(chunks) == 3
    assert all(isinstance(c, ChatCompletionStreamChunk) for c in chunks)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[1].choices[0].delta.content == "Hello"
    assert chunks[2].choices[0].finish_reason == "stop"
