"""Fixtures shared across tests/unit/async_client/."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone.async_client.assistants import AsyncAssistants
from pinecone.models.assistant.model import AssistantModel

BASE_URL = "https://api.test.pinecone.io"

_CANNED_ASSISTANT = AssistantModel(
    name="legacy-name",
    status="Ready",
    created_at="2025-01-15T12:00:00Z",
    updated_at="2025-01-15T12:00:00Z",
    metadata={},
    instructions=None,
    host="test-assistant-abc123.svc.pinecone.io",
)


@pytest.fixture
def mock_async_assistants() -> AsyncAssistants:
    """Return an AsyncAssistants instance whose underlying HTTP client is mocked.

    The mock intercepts every HTTP call so tests exercise only the
    parameter-handling logic in AsyncAssistants.create(), not network I/O.
    """
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    obj = AsyncAssistants(config=config)

    # Stub the HTTP client so no real requests go out.
    mock_http = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = b"{}"
    mock_http.post.return_value = mock_response
    obj._http = mock_http  # type: ignore[attr-defined]

    # Stub the adapter so it returns canned responses.
    mock_adapter = MagicMock()
    mock_adapter.to_assistant.return_value = _CANNED_ASSISTANT
    obj._adapter = mock_adapter  # type: ignore[attr-defined]

    # Stub _poll_until_ready so tests don't loop.
    obj._poll_until_ready = AsyncMock(return_value=_CANNED_ASSISTANT)  # type: ignore[method-assign]

    return obj
