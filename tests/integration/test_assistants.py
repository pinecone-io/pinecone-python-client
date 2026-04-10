"""Integration tests for assistant CRUD lifecycle (sync / REST).

Tests cover the `assistant-lifecycle` area tag:
  - create assistant with name and instructions
  - verify AssistantModel fields (name, status, created_at)
  - describe, list, update (change instructions), delete

These tests make real API calls and require PINECONE_API_KEY in the environment.
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.models.assistant.model import AssistantModel
from tests.integration.conftest import cleanup_resource, unique_name, wait_for_ready


# ---------------------------------------------------------------------------
# assistant-lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "SDK bug IT-0008: Assistants HTTP client uses wrong base path. "
        "SDK posts to /assistants (404) instead of /assistant/assistants. "
        "The Assistants class must prepend /assistant/ to its control-plane routes."
    ),
    strict=True,
)
def test_assistant_lifecycle_create_describe_list_update_delete(client: Pinecone) -> None:
    """Create assistant, verify fields, describe, list, update instructions, delete."""
    name = unique_name("asst")
    try:
        # --- Create ---
        assistant = client.assistants.create(
            name=name,
            instructions="You are a test assistant.",
        )
        assert isinstance(assistant, AssistantModel)
        assert assistant.name == name
        assert assistant.status in ("Initializing", "Ready")
        # created_at may be populated after create or after polling
        # The SDK polls until Ready by default

        # Poll until Ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # --- Describe ---
        described = client.assistants.describe(name=name)
        assert isinstance(described, AssistantModel)
        assert described.name == name
        assert described.status == "Ready"
        assert described.instructions == "You are a test assistant."
        # created_at should be set
        assert described.created_at is not None
        assert isinstance(described.created_at, str)
        assert len(described.created_at) > 0
        # host should be set once Ready
        assert described.host is not None

        # --- List ---
        assistants = client.assistants.list().to_list()
        names = [a.name for a in assistants]
        assert name in names

        # --- Update ---
        updated = client.assistants.update(
            name=name,
            instructions="Updated instructions for the test assistant.",
        )
        assert isinstance(updated, AssistantModel)
        assert updated.name == name
        assert updated.instructions == "Updated instructions for the test assistant."

        # Verify update persisted via describe
        re_described = client.assistants.describe(name=name)
        assert re_described.instructions == "Updated instructions for the test assistant."

    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )
