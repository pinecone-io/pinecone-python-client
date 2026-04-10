"""Integration tests for assistant operations (sync / REST).

Tests cover:
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

from pinecone import Pinecone
from pinecone.models.assistant.chat import ChatResponse
from pinecone.models.assistant.context import ContextResponse
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.model import AssistantModel
from tests.integration.conftest import cleanup_resource, unique_name, wait_for_ready

# ---------------------------------------------------------------------------
# assistant-lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
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


# ---------------------------------------------------------------------------
# assistant-files
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_assistant_files_upload_list_describe_delete(client: Pinecone) -> None:
    """Upload a file to an assistant, list files, describe file, delete file."""
    name = unique_name("asst")
    tmp_path: str | None = None
    file_id: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="Test file assistant.")
        assert isinstance(assistant, AssistantModel)

        # Wait for assistant to be ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
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
        file_model = client.assistants.upload_file(
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
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Describe file — verify fields
        described_file = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        assert isinstance(described_file, AssistantFileModel)
        assert described_file.id == file_id
        assert described_file.name is not None
        assert described_file.status in ("Available", "Processed")

        # List files — verify file appears
        files = client.assistants.list_files(assistant_name=name).to_list()
        file_ids = [f.id for f in files]
        assert file_id in file_ids

        # Delete the file
        client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None  # mark as cleaned

    finally:
        # Clean up temp file
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        # Delete the assistant (also cleans up any remaining files)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_chat_non_streaming_response(client: Pinecone) -> None:
    """Send a non-streaming chat message and verify ChatResponse structure."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small knowledge file so the assistant has context to respond to
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-chat-"
        ) as f:
            f.write("Pinecone is a managed vector database. It supports dense and sparse vectors.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Send a non-streaming chat message
        response = client.assistants.chat(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-context
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_context_retrieval(client: Pinecone) -> None:
    """Retrieve context from an assistant and verify ContextResponse structure."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
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

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Retrieve context using a message
        response = client.assistants.context(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )
