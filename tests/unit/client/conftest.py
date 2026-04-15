"""Fixtures shared across tests/unit/client/."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone.client.assistants import Assistants
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
def mock_assistants() -> Assistants:
    """Return an Assistants instance whose underlying HTTP client is mocked.

    The mock intercepts every HTTP call so tests exercise only the
    parameter-handling logic in Assistants.create(), not network I/O.
    """
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    obj = Assistants(config=config)

    # Stub the HTTP client so no real requests go out.
    obj._http = MagicMock()  # type: ignore[attr-defined]

    # Stub the adapter so it returns a canned AssistantModel.
    mock_adapter = MagicMock()
    mock_adapter.to_assistant.return_value = _CANNED_ASSISTANT
    obj._adapter = mock_adapter  # type: ignore[attr-defined]

    # Stub _poll_until_ready so tests don't loop.
    obj._poll_until_ready = MagicMock(return_value=_CANNED_ASSISTANT)  # type: ignore[method-assign]

    return obj
