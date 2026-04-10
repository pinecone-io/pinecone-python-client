"""Integration tests for assistant operations (async / REST).

Tests cover using AsyncPinecone:
  - `assistant-lifecycle`: create, describe, list, update, delete
  - `assistant-files`: upload file, list, describe file, delete file
  - `assistant-chat`: non-streaming chat, verify ChatResponse structure
  - `assistant-context`: retrieve context snippets, verify ContextResponse

These tests make real API calls and require PINECONE_API_KEY in the environment.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from pinecone import AsyncPinecone
from pinecone.models.assistant.chat import ChatResponse
from pinecone.models.assistant.context import ContextResponse
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.model import AssistantModel
from tests.integration.conftest import async_cleanup_resource, async_poll_until, unique_name

# ---------------------------------------------------------------------------
# assistant-lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
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


# ---------------------------------------------------------------------------
# Shared xfail marker
# ---------------------------------------------------------------------------

_XFAIL_IT0008 = pytest.mark.xfail(
    reason=(
        "SDK bug IT-0008: Assistants HTTP client uses wrong base path. "
        "SDK posts to /assistants (404) instead of /assistant/assistants. "
        "The Assistants class must prepend /assistant/ to its control-plane routes."
    ),
    strict=True,
)


# ---------------------------------------------------------------------------
# assistant-files
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
@_XFAIL_IT0008
async def test_assistant_files_upload_list_describe_delete(
    async_client: AsyncPinecone,
) -> None:
    """Upload a file to an assistant, list files, describe file, delete file."""
    name = unique_name("asst")
    tmp_path: str | None = None
    file_id: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(
            name=name, instructions="Test file assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait until Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Create a small temp text file on disk
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-test-"
        ) as f:
            f.write("Pinecone is a vector database used for building AI applications.")
            tmp_path = f.name

        # Upload the file
        file_model = await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        assert file_model.id is not None
        assert file_model.name is not None
        assert file_model.status in ("Available", "Processing", "Processed")
        file_id = file_model.id

        # Poll until file is processed (Available)
        await async_poll_until(
            lambda: async_client.assistants.describe_file(assistant_name=name, file_id=file_id),
            lambda f: f.status in ("Available", "Processed"),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Describe file — verify fields
        described_file = await async_client.assistants.describe_file(
            assistant_name=name, file_id=file_id
        )
        assert isinstance(described_file, AssistantFileModel)
        assert described_file.id == file_id
        assert described_file.name is not None
        assert described_file.status in ("Available", "Processed")

        # List files — verify file appears
        files = await async_client.assistants.list_files(assistant_name=name).to_list()
        file_ids = [f.id for f in files]
        assert file_id in file_ids

        # Delete the file
        await async_client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None  # mark as cleaned

    finally:
        # Clean up temp file
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        # Delete the assistant
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
@_XFAIL_IT0008
async def test_assistant_chat_non_streaming_response(
    async_client: AsyncPinecone,
) -> None:
    """Send a non-streaming chat message and verify ChatResponse structure."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small knowledge file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-chat-"
        ) as f:
            f.write("Pinecone is a managed vector database. It supports dense and sparse vectors.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Send a non-streaming chat message
        response = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=False,
        )

        # Verify ChatResponse structure
        assert isinstance(response, ChatResponse)
        assert isinstance(response.id, str) and len(response.id) > 0
        assert isinstance(response.model, str) and len(response.model) > 0
        assert hasattr(response, "message")
        assert response.message.role == "assistant"
        assert isinstance(response.message.content, str) and len(response.message.content) > 0
        assert isinstance(response.finish_reason, str)

    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-context
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
@_XFAIL_IT0008
async def test_assistant_context_retrieval(
    async_client: AsyncPinecone,
) -> None:
    """Retrieve context from an assistant and verify ContextResponse structure."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a knowledge file for context retrieval
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-ctx-"
        ) as f:
            f.write("Pinecone is a managed vector database. It provides fast similarity search.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Retrieve context using a message
        response = await async_client.assistants.context(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
        )

        # Verify ContextResponse structure
        assert isinstance(response, ContextResponse)
        assert hasattr(response, "snippets")
        assert isinstance(response.snippets, list)
        assert hasattr(response, "usage")
        assert response.usage is not None

    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )
