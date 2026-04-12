"""Integration tests for assistant operations (sync / REST).

Tests cover:
  - `assistant-lifecycle`: create, describe, list, update, delete
  - `assistant-files`: upload file, list, describe file, delete file
  - `assistant-chat`: non-streaming chat, verify ChatResponse structure
  - `assistant-context`: retrieve context snippets, verify ContextResponse
  - `assistant-chat-streaming`: streaming chat, verify chunk structure
  - `assistant-chat-completions`: OpenAI-compatible chat completions endpoint
  - `assistant-evaluate`: evaluate_alignment, verify AlignmentResult structure

These tests make real API calls and require PINECONE_API_KEY in the environment.
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import pytest

from pinecone import Pinecone, PineconeValueError
from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse
from pinecone.models.assistant.context import ContextResponse
from pinecone.models.assistant.evaluation import AlignmentResult
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.streaming import StreamContentChunk
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
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
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
        assert isinstance(response.id, str)
        assert len(response.id) > 0
        assert isinstance(response.model, str)
        assert len(response.model) > 0
        assert hasattr(response, "message")
        assert response.message.role == "assistant"
        assert isinstance(response.message.content, str)
        assert len(response.message.content) > 0
        assert isinstance(response.finish_reason, str)

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
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
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_chat_streaming_returns_content_chunks(client: Pinecone) -> None:
    """Stream chat(stream=True), verify at least one content chunk with non-empty text."""
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

        # Upload a small knowledge file so the assistant has context
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-stream-"
        ) as f:
            f.write("Pinecone is a managed vector database for AI applications.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Streaming chat — returns Iterator[ChatStreamChunk]
        stream = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=True,
        )

        # Consume the entire stream
        chunks = list(stream)

        # Must have received at least one chunk
        assert len(chunks) > 0, "Expected at least one chunk from streaming chat"

        # At least one chunk must be a StreamContentChunk with non-empty content
        content_chunks = [c for c in chunks if isinstance(c, StreamContentChunk)]
        assert len(content_chunks) > 0, "Expected at least one StreamContentChunk"

        # Concatenated content must be non-empty
        full_content = "".join(c.delta.content for c in content_chunks)
        assert len(full_content) > 0, "Concatenated streaming content must not be empty"

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-completions
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_chat_completions_openai_compatible_response(client: Pinecone) -> None:
    """chat_completions() returns ChatCompletionResponse with OpenAI-compatible structure."""
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

        # Upload a small knowledge file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-cc-"
        ) as f:
            f.write("Pinecone is a managed vector database for similarity search.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Non-streaming chat completions
        response = client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=False,
        )

        # Verify ChatCompletionResponse structure (OpenAI-compatible)
        assert isinstance(response, ChatCompletionResponse)
        assert hasattr(response, "choices")
        assert isinstance(response.choices, list)
        assert len(response.choices) > 0, "Expected at least one choice"

        choice = response.choices[0]
        assert hasattr(choice, "message")
        assert isinstance(choice.message.content, str)
        assert len(choice.message.content) > 0, "Expected non-empty message content"
        assert hasattr(choice, "finish_reason")

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-typed-messages
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_chat_typed_message_objects(client: Pinecone) -> None:
    """Verify chat() accepts typed Message objects, not just plain dicts (unified-chat-0003).

    Sends a multi-turn conversation where both the initial user message and the
    follow-up (including the prior assistant turn) are typed Message objects.
    Confirms that the SDK accepts this input format and returns a valid ChatResponse.
    """
    from pinecone.models.assistant.message import Message

    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-typed-"
        ) as f:
            f.write("Pinecone is a managed vector database. It stores dense and sparse vectors.")
            tmp_path = f.name

        client.assistants.upload_file(assistant_name=name, file_path=tmp_path, timeout=120)

        # --- Turn 1: typed Message object (not a plain dict) ---
        first_message = Message(role="user", content="What is Pinecone?")
        assert isinstance(first_message, Message)

        first_response = client.assistants.chat(
            assistant_name=name,
            messages=[first_message],
            stream=False,
        )
        assert isinstance(first_response, ChatResponse)
        assert first_response.message.role == "assistant"
        assert isinstance(first_response.message.content, str)
        assert len(first_response.message.content) > 0

        # --- Turn 2: full typed-object history (user + assistant + user) ---
        history = [
            first_message,
            Message(role="assistant", content=first_response.message.content),
            Message(role="user", content="What vectors does it support?"),
        ]
        assert all(isinstance(m, Message) for m in history)

        second_response = client.assistants.chat(
            assistant_name=name,
            messages=history,
            stream=False,
        )
        assert isinstance(second_response, ChatResponse)
        assert second_response.message.role == "assistant"
        assert isinstance(second_response.message.content, str)
        assert len(second_response.message.content) > 0
        # Response IDs must be unique across turns
        assert second_response.id != first_response.id

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-evaluate
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_evaluate_alignment_scores(client: Pinecone) -> None:
    """evaluate_alignment() returns AlignmentResult with correctness/completeness scores."""
    # evaluate_alignment is stateless — no assistant or file needed
    result = client.assistants.evaluate_alignment(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        ground_truth_answer="The capital of France is Paris.",
    )

    # Verify AlignmentResult structure
    assert isinstance(result, AlignmentResult)
    assert hasattr(result, "scores")
    assert hasattr(result.scores, "alignment")
    assert hasattr(result.scores, "correctness")
    assert hasattr(result.scores, "completeness")

    # Scores should be floats in [0, 1]
    assert isinstance(result.scores.alignment, float)
    assert isinstance(result.scores.correctness, float)
    assert isinstance(result.scores.completeness, float)
    assert 0.0 <= result.scores.alignment <= 1.0
    assert 0.0 <= result.scores.correctness <= 1.0
    assert 0.0 <= result.scores.completeness <= 1.0

    # A well-aligned answer should score high
    assert result.scores.alignment > 0.5, f"Expected high alignment, got {result.scores.alignment}"

    # facts list and usage should be present
    assert hasattr(result, "facts")
    assert isinstance(result.facts, list)
    assert hasattr(result, "usage")
    assert result.usage is not None


# ---------------------------------------------------------------------------
# assistant-validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_create_region_validation_and_chat_stream_json_conflict(
    client: Pinecone,
) -> None:
    """Region validation and stream+json_response mutual exclusivity fire before any API call.

    Verifies:
    - unified-assistant-0013: create() raises PineconeValueError for regions other than
      "us" or "eu" (e.g. "au", "invalid")
    - unified-assistant-0015: Region validation is case-sensitive — "US" and "EU" are
      rejected; only lowercase "us" and "eu" are valid
    - unified-chat-0012: chat() raises PineconeValueError when stream=True and
      json_response=True are specified together, before any API call is made

    No real assistant is created. All validations are client-side, firing before
    any HTTP request.
    """
    # unified-assistant-0013: invalid region raises PineconeValueError
    with pytest.raises(PineconeValueError):
        client.assistants.create(name="validation-test", region="au")

    with pytest.raises(PineconeValueError):
        client.assistants.create(name="validation-test", region="invalid-region")

    # unified-assistant-0015: case-sensitive — uppercase "US" and "EU" are rejected
    with pytest.raises(PineconeValueError):
        client.assistants.create(name="validation-test", region="US")

    with pytest.raises(PineconeValueError):
        client.assistants.create(name="validation-test", region="EU")

    # unified-chat-0012: stream=True + json_response=True raises PineconeValueError
    # before any network call; assistant_name does not need to exist
    with pytest.raises(PineconeValueError):
        client.assistants.chat(
            assistant_name="does-not-matter",
            messages=[{"content": "test query"}],
            stream=True,
            json_response=True,
        )
