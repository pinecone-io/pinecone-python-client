"""Integration tests for assistant CRUD lifecycle (async / REST).

Tests cover the `assistant-lifecycle` area tag using AsyncPinecone:
  - create assistant with name and instructions
  - verify AssistantModel fields (name, status, created_at)
  - describe, list, update (change instructions), delete

These tests make real API calls and require PINECONE_API_KEY in the environment.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from pinecone import AsyncPinecone
from pinecone.models.assistant.model import AssistantModel
from tests.integration.conftest import async_cleanup_resource, async_poll_until, unique_name


# ---------------------------------------------------------------------------
# assistant-lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "SDK bug IT-0008: Assistants HTTP client uses wrong base path. "
        "SDK posts to /assistants (404) instead of /assistant/assistants. "
        "The Assistants class must prepend /assistant/ to its control-plane routes."
    ),
    strict=True,
)
async def test_assistant_lifecycle_create_describe_list_update_delete(
    async_client: AsyncPinecone,
) -> None:
    """Create assistant, verify fields, describe, list, update instructions, delete."""
    name = unique_name("asst")
    try:
        # --- Create ---
        assistant = await async_client.assistants.create(
            name=name,
            instructions="You are an async test assistant.",
        )
        assert isinstance(assistant, AssistantModel)
        assert assistant.name == name
        assert assistant.status in ("Initializing", "Ready")

        # Poll until Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # --- Describe ---
        described = await async_client.assistants.describe(name=name)
        assert isinstance(described, AssistantModel)
        assert described.name == name
        assert described.status == "Ready"
        assert described.instructions == "You are an async test assistant."
        assert described.created_at is not None
        assert isinstance(described.created_at, str)
        assert len(described.created_at) > 0
        assert described.host is not None

        # --- List ---
        assistants = await async_client.assistants.list().to_list()
        names = [a.name for a in assistants]
        assert name in names

        # --- Update ---
        updated = await async_client.assistants.update(
            name=name,
            instructions="Updated async instructions for the test assistant.",
        )
        assert isinstance(updated, AssistantModel)
        assert updated.name == name
        assert updated.instructions == "Updated async instructions for the test assistant."

        # Verify update persisted via describe
        re_described = await async_client.assistants.describe(name=name)
        assert re_described.instructions == "Updated async instructions for the test assistant."

    finally:
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )
