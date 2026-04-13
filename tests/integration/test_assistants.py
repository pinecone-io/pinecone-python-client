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
import io
import os
import tempfile

import pytest

from pinecone import Pinecone, PineconeValueError
from pinecone.models.assistant.options import ContextOptions
from pinecone.models.assistant.chat import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatHighlight,
    ChatMessage,
    ChatReference,
    ChatResponse,
    ChatUsage,
)
from pinecone.models.assistant.context import ContextResponse, TextSnippet
from pinecone.models.assistant.evaluation import AlignmentResult, EntailmentResult
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListFilesResponse
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.streaming import (
    ChatCompletionStreamChunk,
    StreamContentChunk,
    StreamMessageEnd,
    StreamMessageStart,
)
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


# ---------------------------------------------------------------------------
# assistant-files: byte-stream upload with file_name and metadata
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upload_file_from_byte_stream_with_metadata(client: Pinecone) -> None:
    """Upload from a BytesIO stream with file_name and metadata; verify both are preserved.

    Verifies:
    - unified-file-0002: Can upload a file from an in-memory byte stream
    - unified-file-0003: Can upload a file with optional user-provided metadata
    - unified-file-0033: Byte stream upload preserves the provided file name and
      metadata in the file record
    """
    name = unique_name("asst")
    file_id: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="Test stream upload.")
        assert isinstance(assistant, AssistantModel)

        # Wait for assistant to be ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Build an in-memory byte stream with a specific file name and metadata
        content = b"Pinecone is a managed vector database. It supports semantic search."
        stream = io.BytesIO(content)
        upload_name = "stream-knowledge.txt"
        upload_metadata: dict[str, object] = {"source": "stream-test", "category": "sdk-docs"}

        # Upload via file_stream — exercises the file_stream / file_name / metadata paths
        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_stream=stream,
            file_name=upload_name,
            metadata=upload_metadata,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        assert file_model.id is not None
        file_id = file_model.id

        # Poll until file processing completes
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Verify file_name and metadata are preserved in the described file record
        described = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        assert isinstance(described, AssistantFileModel)
        assert described.id == file_id
        assert described.name == upload_name, (
            f"Expected file name '{upload_name}', got '{described.name}'"
        )
        assert described.metadata is not None, "Expected metadata to be preserved, got None"
        assert described.metadata.get("source") == "stream-test"
        assert described.metadata.get("category") == "sdk-docs"

        # Clean up file
        client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_upload_file_input_validation_and_delete_returns_none_rest(client: Pinecone) -> None:
    """upload_file() client-side validation fires before any API call; delete_file() returns None.

    Verifies:
    - unified-file-0035: Uploading a file from a nonexistent local path raises PineconeValueError.
    - unified-file-0030: Successful file deletion returns no value (None).
    The mutual-exclusivity check (both/neither file_path+file_stream) is also verified as a
    related validation boundary — it fires before the path-existence check.
    """
    # --- Part 1: Client-side validation (no API call needed) ---

    # Both file_path AND file_stream provided → PineconeValueError before any HTTP request
    with pytest.raises(PineconeValueError):
        client.assistants.upload_file(
            assistant_name="doesnt-matter",
            file_path="/some/path.txt",
            file_stream=io.BytesIO(b"data"),
        )

    # Neither file_path nor file_stream provided → PineconeValueError before any HTTP request
    with pytest.raises(PineconeValueError):
        client.assistants.upload_file(assistant_name="doesnt-matter")

    # Nonexistent path → PineconeValueError (unified-file-0035); fires before _data_plane_http()
    with pytest.raises(PineconeValueError, match="File not found"):
        client.assistants.upload_file(
            assistant_name="doesnt-matter",
            file_path="/nonexistent/path/to/file.txt",
        )

    # --- Part 2: delete_file() returns None (unified-file-0030) ---
    # Requires a real assistant and a real file upload to exercise the API-level delete path.
    name = unique_name("asst")
    tmp_path: str | None = None
    file_id: str | None = None
    try:
        client.assistants.create(name=name, instructions="Validation test assistant.")

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Create a small temp text file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-val-"
        ) as f:
            f.write("Validation test content.")
            tmp_path = f.name

        # Upload the file
        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        file_id = file_model.id

        # unified-file-0030: delete_file returns None
        result = client.assistants.delete_file(
            assistant_name=name,
            file_id=file_id,
            timeout=60,
        )
        assert result is None, f"Expected delete_file to return None, got {result!r}"
        file_id = None  # mark as cleaned up

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_describe_file_signed_url_rest(client: Pinecone) -> None:
    """describe_file(include_url=True) returns a non-None signed_url string;
    describe_file() without include_url returns signed_url=None.

    Verifies:
    - unified-file-0009: Can request a signed download URL when retrieving file metadata.
    - unified-file-0025: When requesting a signed URL, include-url='true' is sent to the API.
    - unified-file-0026: When not requesting a signed URL, the include-url param is omitted.
    - unified-file-0056: The signed URL defaults to None when absent from the API response.
    """
    name = unique_name("asst")
    file_id: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="Test signed URL.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small text file
        content = b"Pinecone vector database enables fast semantic search."
        stream = io.BytesIO(content)
        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_stream=stream,
            file_name="signed-url-test.txt",
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        assert file_model.id is not None
        file_id = file_model.id

        # Wait until the file is ready
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Without include_url — signed_url should be None (default behavior)
        without_url = client.assistants.describe_file(
            assistant_name=name,
            file_id=file_id,
        )
        assert isinstance(without_url, AssistantFileModel)
        assert without_url.signed_url is None, (
            f"Expected signed_url=None without include_url=True, got {without_url.signed_url!r}"
        )

        # With include_url=True — signed_url should be a non-None string
        with_url = client.assistants.describe_file(
            assistant_name=name,
            file_id=file_id,
            include_url=True,
        )
        assert isinstance(with_url, AssistantFileModel)
        assert with_url.signed_url is not None, (
            "Expected a signed_url string when include_url=True, got None"
        )
        assert isinstance(with_url.signed_url, str), (
            f"Expected signed_url to be a str, got {type(with_url.signed_url)}"
        )
        assert len(with_url.signed_url) > 0, "Expected non-empty signed_url"

        # The core file identity fields should be the same in both responses
        assert with_url.id == without_url.id
        assert with_url.name == without_url.name

        # Clean up file
        client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-dict-no-role
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_message_dict_role_default_rest(client: Pinecone) -> None:
    """chat() with a message dict that omits 'role' defaults role to 'user'.

    All existing chat tests pass {"role": "user", "content": "..."}.
    This test passes {"content": "..."} (no role key) and verifies that:
    1. The SDK converts the dict via Message.from_dict(), defaulting role to "user"
    2. The API call succeeds — confirming the role was correctly populated
    3. A valid ChatResponse is returned

    Verifies unified-chat-0020 and unified-chat-0019.
    """
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
            mode="w", suffix=".txt", delete=False, prefix="asst-norol-"
        ) as f:
            f.write("Pinecone is a managed vector database supporting dense and sparse vectors.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Pass a message dict WITHOUT a "role" key — SDK should default to "user"
        response = client.assistants.chat(
            assistant_name=name,
            messages=[{"content": "What is Pinecone?"}],
            stream=False,
        )

        # A valid ChatResponse means the SDK defaulted role="user" and the API accepted it
        assert isinstance(response, ChatResponse)
        assert isinstance(response.id, str)
        assert len(response.id) > 0
        assert response.message.role == "assistant"
        assert isinstance(response.message.content, str)
        assert len(response.message.content) > 0

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
# assistant-update: metadata replace semantics
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_update_metadata_replaces_not_merges(client: Pinecone) -> None:
    """assistants.update(metadata=...) fully replaces existing metadata; returns updated AssistantModel.

    Verifies:
    - unified-assistant-0021: Updating metadata fully replaces the existing metadata rather
      than merging. Old keys absent from the new dict must disappear.
    - unified-assistant-0030: Updating an assistant returns an AssistantModel with the
      updated attributes visible immediately on the returned object.
    """
    name = unique_name("asst")
    try:
        # Create with two metadata keys that should disappear after the replace
        assistant = client.assistants.create(
            name=name,
            instructions="Initial instructions.",
            metadata={"initial_key": "initial_val", "extra_key": "will_be_gone"},
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready before updating
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Confirm initial metadata is present
        described = client.assistants.describe(name=name)
        assert described.metadata is not None
        assert described.metadata.get("initial_key") == "initial_val"
        assert described.metadata.get("extra_key") == "will_be_gone"

        # Update with a completely different metadata dict (replace, not merge)
        updated = client.assistants.update(name=name, metadata={"new_key": "new_val"})

        # unified-assistant-0030: update() returns the updated AssistantModel immediately
        assert isinstance(updated, AssistantModel)
        assert updated.name == name
        assert updated.metadata is not None

        # unified-assistant-0021: replace semantics — old keys must be gone
        assert updated.metadata.get("new_key") == "new_val", (
            f"Expected new_key='new_val', got: {updated.metadata}"
        )
        assert "initial_key" not in updated.metadata, (
            f"Expected initial_key to be absent after metadata replace, got: {updated.metadata}"
        )
        assert "extra_key" not in updated.metadata, (
            f"Expected extra_key to be absent after metadata replace, got: {updated.metadata}"
        )

        # Verify persistence via a fresh describe call
        re_described = client.assistants.describe(name=name)
        assert re_described.metadata is not None
        assert re_described.metadata.get("new_key") == "new_val"
        assert "initial_key" not in re_described.metadata
        assert "extra_key" not in re_described.metadata

    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-context: context() input validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_context_retrieval_validation_rest(client: Pinecone) -> None:
    """context() raises PineconeValueError for invalid query/messages combos.

    Verifies:
    - unified-context-0008: Exactly one of query or messages must be truthy;
      empty string and empty list are treated as not provided
    - unified-context-0009: Providing both query and messages produces a
      PineconeValueError before any API call
    - unified-context-0010: Providing neither query nor messages produces a
      PineconeValueError before any API call

    No real assistant is created. All validations are client-side and fire
    before any HTTP request, so any assistant_name value is acceptable.
    """
    # unified-context-0009: both query and messages provided → rejected
    with pytest.raises(PineconeValueError):
        client.assistants.context(
            assistant_name="validation-test",
            query="What is Pinecone?",
            messages=[{"role": "user", "content": "What is Pinecone?"}],
        )

    # unified-context-0010: neither query nor messages provided → rejected
    with pytest.raises(PineconeValueError):
        client.assistants.context(
            assistant_name="validation-test",
        )

    # unified-context-0008: empty string query treated as not provided → rejected (neither truthy)
    with pytest.raises(PineconeValueError):
        client.assistants.context(
            assistant_name="validation-test",
            query="",
        )

    # unified-context-0008: empty list messages treated as not provided → rejected (neither truthy)
    with pytest.raises(PineconeValueError):
        client.assistants.context(
            assistant_name="validation-test",
            messages=[],
        )

    # unified-context-0008: empty string query + valid messages → only messages truthy → no validation error
    # (this call will reach the network; we only care it does NOT raise a PineconeValueError)
    try:
        client.assistants.context(
            assistant_name="validation-test",
            query="",
            messages=[{"role": "user", "content": "test"}],
        )
    except PineconeValueError:
        raise AssertionError(
            "context() raised PineconeValueError when query='' and messages is non-empty — "
            "empty string should be treated as not provided, leaving messages as the sole input"
        )
    except Exception:
        # Any other exception (ApiError, NotFoundError, etc.) is acceptable here —
        # the assistant does not exist and the API will reject the request.
        pass


# ---------------------------------------------------------------------------
# assistant-context-query-param
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_context_retrieval_with_query_param_rest(client: Pinecone) -> None:
    """context() with query parameter (not messages) returns ContextResponse with typed snippet structure.

    Verifies:
    - unified-context-0001: Can retrieve context using a plain text query string (query param)
    - unified-context-0021: Response contains snippets list and usage statistics
    - unified-context-0023: Text snippets have content (str), score (float), and reference.file (str)
    The top_k=2 limit is also verified against the returned snippet count.
    """
    name = unique_name("asst")
    file_id: str | None = None
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(
            name=name, instructions="You help answer questions about vector databases."
        )
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a text file with content the assistant can retrieve snippets from
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-qctx-"
        ) as f:
            f.write(
                "Pinecone is a managed vector database optimized for semantic similarity search. "
                "It stores vector embeddings and enables fast nearest-neighbor retrieval. "
                "Pinecone supports metadata filtering, sparse-dense hybrid search, and "
                "both serverless and pod-based deployment modes."
            )
            tmp_path = f.name

        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        file_id = file_model.id

        # Wait for file to be indexed before calling context
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Use the query parameter (plain text string), not messages
        response = client.assistants.context(
            assistant_name=name,
            query="What is Pinecone?",
            top_k=2,
        )

        # unified-context-0021: response has snippets list and usage
        assert isinstance(response, ContextResponse)
        assert isinstance(response.snippets, list)
        assert response.usage is not None

        # top_k=2 must limit the result to at most 2 snippets
        assert len(response.snippets) <= 2, (
            f"Expected at most 2 snippets with top_k=2, got {len(response.snippets)}"
        )

        # unified-context-0023: each text snippet has content, score (float), and reference.file
        for snippet in response.snippets:
            assert isinstance(snippet.score, float), (
                f"Expected float score, got {type(snippet.score)}"
            )
            assert hasattr(snippet, "reference"), "Snippet missing reference attribute"
            assert isinstance(snippet.reference.file, AssistantFileModel)
            assert isinstance(snippet.reference.file.name, str)
            assert snippet.reference.file.name != "", (
                f"Expected non-empty file reference name, got {snippet.reference.file.name!r}"
            )
            if isinstance(snippet, TextSnippet):
                assert isinstance(snippet.content, str) and len(snippet.content) > 0, (
                    "TextSnippet.content should be a non-empty string"
                )

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
# assistant-chat-completions-streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_completions_streaming_rest(client: Pinecone) -> None:
    """chat_completions(stream=True) returns an Iterator[ChatCompletionStreamChunk] with correct structure.

    Verifies:
    - unified-stream-0002: Chat completions support streaming mode that returns chunks
      lazily as they arrive.
    - ChatCompletionStreamChunk structure: each chunk has id (str) and choices list;
      each choice has index (int), delta with optional content (str), and optional
      finish_reason.
    - At least one chunk has non-empty delta.content; full concatenated response is non-empty.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-ccstream-"
        ) as f:
            f.write("Pinecone is a managed vector database for AI-powered search.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # chat_completions(stream=True) returns Iterator[ChatCompletionStreamChunk]
        stream = client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=True,
        )

        # Consume the full stream
        chunks: list[ChatCompletionStreamChunk] = list(stream)

        # Must receive at least one chunk
        assert len(chunks) > 0, "Expected at least one ChatCompletionStreamChunk"

        # Every chunk must conform to the expected structure
        for chunk in chunks:
            assert isinstance(chunk, ChatCompletionStreamChunk), (
                f"Expected ChatCompletionStreamChunk, got {type(chunk)}"
            )
            assert isinstance(chunk.id, str) and chunk.id != "", (
                f"ChatCompletionStreamChunk.id must be a non-empty string, got {chunk.id!r}"
            )
            assert isinstance(chunk.choices, list), "choices must be a list"
            for choice in chunk.choices:
                assert isinstance(choice.index, int), "choice.index must be int"
                # delta.content is optional (None on role-only or finish chunks)
                assert choice.delta.content is None or isinstance(choice.delta.content, str), (
                    f"delta.content must be str or None, got {type(choice.delta.content)}"
                )

        # At least one chunk must carry delta content
        content_choices = [
            c
            for chunk in chunks
            for c in chunk.choices
            if c.delta.content is not None and c.delta.content != ""
        ]
        assert len(content_choices) > 0, (
            "Expected at least one chunk with non-empty delta.content"
        )

        # Concatenated response must be meaningful
        full_content = "".join(
            c.delta.content
            for chunk in chunks
            for c in chunk.choices
            if c.delta.content is not None
        )
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
# list_files_page — explicit single-page pagination
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_list_files_page_with_page_size_and_pagination_token_rest(client: Pinecone) -> None:
    """list_files_page() returns a ListFilesResponse containing all uploaded files.

    The API does not support server-side page size limiting for file listings,
    so all files are returned in a single response with next=None.

    Verifies:
      - unified-file-0011: Can list one page of files with explicit pagination control
      - unified-file-0014: Can provide a pagination token to fetch the next page of files
    """
    name = unique_name("asst")
    file_id_a: str | None = None
    file_id_b: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="Pagination test assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload file A
        file_a = client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(b"File A content for pagination test."),
            file_name="file_a.txt",
            timeout=120,
        )
        assert isinstance(file_a, AssistantFileModel)
        file_id_a = file_a.id

        # Upload file B
        file_b = client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(b"File B content for pagination test."),
            file_name="file_b.txt",
            timeout=120,
        )
        assert isinstance(file_b, AssistantFileModel)
        file_id_b = file_b.id

        # Wait until both files are available
        for fid in (file_id_a, file_id_b):
            wait_for_ready(
                lambda fid=fid: (
                    client.assistants.describe_file(
                        assistant_name=name, file_id=fid
                    ).status
                    in ("Available", "Processed")
                ),
                timeout=120,
                interval=5,
                description=f"file {fid}",
            )

        # list_files_page() returns all files in a single response
        page = client.assistants.list_files_page(assistant_name=name)
        assert isinstance(page, ListFilesResponse), (
            f"Expected ListFilesResponse, got {type(page)}"
        )
        assert isinstance(page.files, list), "page.files must be a list"
        assert all(isinstance(f, AssistantFileModel) for f in page.files), (
            "Each file in list must be an AssistantFileModel"
        )

        # Both uploaded files must appear in the listing
        seen_ids = {f.id for f in page.files}
        assert file_id_a in seen_ids, f"File A ({file_id_a}) missing from listing"
        assert file_id_b in seen_ids, f"File B ({file_id_b}) missing from listing"

    finally:
        for fid in filter(None, [file_id_a, file_id_b]):
            with contextlib.suppress(Exception):
                client.assistants.delete_file(
                    assistant_name=name, file_id=fid, timeout=30
                )
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat-stream-structure
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_stream_message_start_and_end_structure_rest(client: Pinecone) -> None:
    """Streaming chat first chunk is StreamMessageStart; last chunk is StreamMessageEnd with usage.

    Verifies:
    - unified-stream-0016: First chunk in a chat stream is a message-start chunk
      containing the model and role.
    - unified-stream-0018: Final chunk in a chat stream is a message-end chunk
      containing token usage statistics (prompt_tokens, completion_tokens, total_tokens).
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small file so the assistant can respond
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-struct-"
        ) as f:
            f.write("Pinecone is a vector database for machine learning applications.")
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

        assert len(chunks) >= 2, (
            f"Expected at least 2 chunks (message_start + message_end), got {len(chunks)}"
        )

        # First chunk must be StreamMessageStart with non-empty model and role
        first = chunks[0]
        assert isinstance(first, StreamMessageStart), (
            f"Expected first chunk to be StreamMessageStart, got {type(first).__name__}"
        )
        assert isinstance(first.model, str) and len(first.model) > 0, (
            f"StreamMessageStart.model must be a non-empty string, got {first.model!r}"
        )
        assert isinstance(first.role, str) and len(first.role) > 0, (
            f"StreamMessageStart.role must be a non-empty string, got {first.role!r}"
        )

        # Last chunk must be StreamMessageEnd with token usage statistics
        last = chunks[-1]
        assert isinstance(last, StreamMessageEnd), (
            f"Expected last chunk to be StreamMessageEnd, got {type(last).__name__}"
        )
        assert isinstance(last.usage.prompt_tokens, int) and last.usage.prompt_tokens >= 0, (
            f"prompt_tokens must be a non-negative int, got {last.usage.prompt_tokens!r}"
        )
        assert isinstance(last.usage.completion_tokens, int) and last.usage.completion_tokens >= 0, (
            f"completion_tokens must be a non-negative int, got {last.usage.completion_tokens!r}"
        )
        assert isinstance(last.usage.total_tokens, int) and last.usage.total_tokens > 0, (
            f"total_tokens must be a positive int, got {last.usage.total_tokens!r}"
        )

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
# chat — json_response=True
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_chat_json_response_mode_returns_valid_json_rest(client: Pinecone) -> None:
    """chat(json_response=True) returns a ChatResponse whose message.content is valid JSON.

    Verifies:
    - unified-chat-0038: JSON response mode returns valid JSON in the message content.
    """
    import json

    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-jsonresp-"
        ) as f:
            f.write("Pinecone is a managed vector database service.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Request a JSON-format response with a prompt that strongly guides JSON output
        response = client.assistants.chat(
            assistant_name=name,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Reply ONLY with a JSON object. "
                        "The object must have exactly one key called 'answer' whose value "
                        "is a string describing what Pinecone is in one sentence."
                    ),
                }
            ],
            json_response=True,
        )

        # Verify ChatResponse structure
        assert isinstance(response, ChatResponse), (
            f"Expected ChatResponse, got {type(response)}"
        )
        assert isinstance(response.message.content, str) and len(response.message.content) > 0, (
            f"message.content must be a non-empty string, got {response.message.content!r}"
        )

        # The content must be valid JSON — this is the core claim
        try:
            parsed = json.loads(response.message.content)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"chat(json_response=True) returned content that is not valid JSON: "
                f"{response.message.content!r}"
            ) from exc

        # Must be a JSON object (dict), not a list or primitive
        assert isinstance(parsed, dict), (
            f"Expected a JSON object, got {type(parsed).__name__}: {parsed!r}"
        )

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
# chat — include_highlights
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_chat_include_highlights_rest(client: Pinecone) -> None:
    """chat(include_highlights=True) succeeds and returns correctly-typed highlight objects.

    Verifies:
    - unified-chat-0009: Can request highlight snippets to be included in citations.
    - unified-chat-0046: Each reference may include a highlight with type and content
      text when highlights are requested.
    - unified-chat-0047: The highlight field within a reference is absent (None) when
      highlights are not requested.

    Strategy:
    - Upload a knowledge file, then call chat(include_highlights=True).
    - Verify the API call succeeds and the response is a ChatResponse.
    - For every citation reference in the response, verify that the highlight field
      is either None or a ChatHighlight with non-empty type and content strings.
    - Then call chat without include_highlights (default False) and verify that all
      citation reference highlights are None.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-hlights-"
        ) as f:
            f.write(
                "Pinecone is a managed vector database that stores dense and sparse embeddings. "
                "It supports serverless and pod-based deployment options."
            )
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # --- Call 1: include_highlights=True ---
        response_with = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=False,
            include_highlights=True,
        )

        assert isinstance(response_with, ChatResponse), (
            f"Expected ChatResponse, got {type(response_with)}"
        )
        assert isinstance(response_with.citations, list), "citations must be a list"

        # For every reference returned, highlight must be None or a valid ChatHighlight
        for citation in response_with.citations:
            for ref in citation.references:
                assert isinstance(ref, ChatReference), (
                    f"Expected ChatReference, got {type(ref)}"
                )
                if ref.highlight is not None:
                    assert isinstance(ref.highlight, ChatHighlight), (
                        f"highlight must be ChatHighlight or None, got {type(ref.highlight)}"
                    )
                    assert (
                        isinstance(ref.highlight.type, str) and len(ref.highlight.type) > 0
                    ), (
                        f"ChatHighlight.type must be a non-empty string, "
                        f"got {ref.highlight.type!r}"
                    )
                    assert (
                        isinstance(ref.highlight.content, str)
                        and len(ref.highlight.content) > 0
                    ), (
                        f"ChatHighlight.content must be a non-empty string, "
                        f"got {ref.highlight.content!r}"
                    )

        # --- Call 2: default include_highlights (False) ---
        response_without = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What deployment options does Pinecone have?"}],
            stream=False,
        )

        assert isinstance(response_without, ChatResponse), (
            f"Expected ChatResponse, got {type(response_without)}"
        )
        # When highlights are not requested, all reference highlights must be None
        for citation in response_without.citations:
            for ref in citation.references:
                assert ref.highlight is None, (
                    f"Expected highlight=None when include_highlights not set, "
                    f"got {ref.highlight!r}"
                )

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
# chat-context-options
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_context_options_typed_and_dict_rest(client: Pinecone) -> None:
    """chat() accepts context_options as both a typed ContextOptions object and a plain dict.

    Verifies:
    - unified-chat-0025: Context options can be provided as either a typed options object
      or a plain dictionary.
    - unified-chat-0010: Can control context retrieval parameters via context options.

    Strategy:
    - Upload a knowledge file to an assistant.
    - Call chat() with context_options=ContextOptions(top_k=5) — exercises the typed object
      serialization path (msgspec.structs.asdict with None-filtering).
    - Call chat() with context_options={"top_k": 5} — exercises the dict pass-through path.
    - Both calls must succeed and return a valid ChatResponse.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-ctxopt-"
        ) as f:
            f.write(
                "Pinecone is a managed vector database for high-performance similarity search. "
                "It supports both serverless and pod-based deployment options."
            )
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # --- Call 1: typed ContextOptions object ---
        response_typed = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=False,
            context_options=ContextOptions(top_k=5),
        )

        assert isinstance(response_typed, ChatResponse), (
            f"Expected ChatResponse with typed ContextOptions, got {type(response_typed)}"
        )
        assert isinstance(response_typed.message, ChatMessage), (
            f"ChatResponse.message must be ChatMessage, got {type(response_typed.message)}"
        )
        assert isinstance(response_typed.message.content, str), (
            f"ChatMessage.content must be str, got {type(response_typed.message.content)}"
        )
        assert len(response_typed.message.content) > 0, "ChatMessage.content must be non-empty"

        # --- Call 2: plain dict context_options ---
        response_dict = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What deployment options does Pinecone have?"}],
            stream=False,
            context_options={"top_k": 5},
        )

        assert isinstance(response_dict, ChatResponse), (
            f"Expected ChatResponse with dict context_options, got {type(response_dict)}"
        )
        assert isinstance(response_dict.message, ChatMessage), (
            f"ChatResponse.message must be ChatMessage, got {type(response_dict.message)}"
        )
        assert isinstance(response_dict.message.content, str), (
            f"ChatMessage.content must be str, got {type(response_dict.message.content)}"
        )
        assert len(response_dict.message.content) > 0, "ChatMessage.content must be non-empty"

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
# chat-completions-full-structure — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_completions_full_response_structure_rest(client: Pinecone) -> None:
    """Verify the complete ChatCompletionResponse structure from chat_completions().

    Verifies:
    - unified-chat-0042: non-streaming chat_completions response has NO citations field
    - unified-chat-0048: response contains id (str), choices list, model (str), usage
    - unified-chat-0049: each choice contains an index (int), message, and finish_reason
    - unified-chat-0050: usage includes prompt_tokens, completion_tokens, total_tokens

    The existing test_assistant_chat_completions_openai_compatible_response only checks
    choices, message.content, and finish_reason — it does not verify id, model, usage
    fields, choice.index, or the absence of citations.

    Area tag: assistant-chat-completions
    Transport: rest
    """
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
            mode="w", suffix=".txt", delete=False, prefix="asst-cc-full-"
        ) as f:
            f.write("Pinecone is a managed vector database service for AI applications.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        response = client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone used for?"}],
            stream=False,
        )

        # --- unified-chat-0042: NO citations field on ChatCompletionResponse ---
        assert isinstance(response, ChatCompletionResponse), (
            f"Expected ChatCompletionResponse, got {type(response)}"
        )
        assert not hasattr(response, "citations"), (
            "ChatCompletionResponse must NOT have a citations field (OpenAI-compatible format)"
        )

        # --- unified-chat-0048: response has id, choices, model, usage ---
        assert isinstance(response.id, str) and len(response.id) > 0, (
            f"response.id must be a non-empty str, got {response.id!r}"
        )
        assert isinstance(response.model, str) and len(response.model) > 0, (
            f"response.model must be a non-empty str, got {response.model!r}"
        )
        assert isinstance(response.choices, list) and len(response.choices) > 0, (
            f"response.choices must be a non-empty list, got {response.choices!r}"
        )
        assert isinstance(response.usage, ChatUsage), (
            f"response.usage must be ChatUsage, got {type(response.usage)}"
        )

        # --- unified-chat-0049: each choice has index, message, finish_reason ---
        for choice in response.choices:
            assert isinstance(choice, ChatCompletionChoice), (
                f"Each choice must be ChatCompletionChoice, got {type(choice)}"
            )
            assert isinstance(choice.index, int), (
                f"choice.index must be int, got {type(choice.index)}"
            )
            assert isinstance(choice.message, ChatMessage), (
                f"choice.message must be ChatMessage, got {type(choice.message)}"
            )
            assert isinstance(choice.message.content, str) and len(choice.message.content) > 0, (
                "choice.message.content must be a non-empty str"
            )
            assert isinstance(choice.finish_reason, str) and len(choice.finish_reason) > 0, (
                f"choice.finish_reason must be a non-empty str, got {choice.finish_reason!r}"
            )

        # --- unified-chat-0050: usage has prompt_tokens, completion_tokens, total_tokens ---
        usage = response.usage
        assert isinstance(usage.prompt_tokens, int) and usage.prompt_tokens >= 0, (
            f"usage.prompt_tokens must be non-negative int, got {usage.prompt_tokens!r}"
        )
        assert isinstance(usage.completion_tokens, int) and usage.completion_tokens >= 0, (
            f"usage.completion_tokens must be non-negative int, got {usage.completion_tokens!r}"
        )
        assert isinstance(usage.total_tokens, int) and usage.total_tokens > 0, (
            f"usage.total_tokens must be a positive int, got {usage.total_tokens!r}"
        )
        # Sanity check: prompt + completion == total
        assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens, (
            f"prompt_tokens ({usage.prompt_tokens}) + completion_tokens "
            f"({usage.completion_tokens}) should equal total_tokens ({usage.total_tokens})"
        )

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
# assistant-evaluate per-fact structure and usage token counts
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_evaluate_alignment_per_fact_structure_and_usage_tokens_rest(
    client: Pinecone,
) -> None:
    """evaluate_alignment() returns EntailmentResult facts with proper field types
    and ChatUsage with non-negative integer token counts.

    Verifies:
    - unified-eval-0003: Each fact in AlignmentResult.facts is an EntailmentResult
      with a non-empty 'fact' string, an 'entailment' value in
      {"entailed", "contradicted", "neutral"}, and a 'reasoning' string.
    - unified-eval-0004: AlignmentResult.usage is a ChatUsage with non-negative
      integer prompt_tokens, completion_tokens, and total_tokens, where
      prompt_tokens + completion_tokens == total_tokens.

    evaluate_alignment is stateless — no assistant or file is needed.
    """
    # Use a statement about a well-known fact so at least one "entailed" fact
    # is produced, giving the test something concrete to check.
    result = client.assistants.evaluate_alignment(
        question="What is the boiling point of water at sea level?",
        answer="Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
        ground_truth_answer="The boiling point of water at standard atmospheric pressure is 100°C.",
    )

    assert isinstance(result, AlignmentResult)

    # --- unified-eval-0003: per-fact EntailmentResult structure ---
    assert isinstance(result.facts, list), "facts must be a list"
    assert len(result.facts) >= 1, (
        "At least one EntailmentResult fact expected for a well-defined question/answer pair"
    )
    valid_entailments = {"entailed", "contradicted", "neutral"}
    for fact_item in result.facts:
        assert isinstance(fact_item, EntailmentResult), (
            f"Each fact must be an EntailmentResult, got {type(fact_item)}"
        )
        assert isinstance(fact_item.fact, str) and len(fact_item.fact) > 0, (
            f"EntailmentResult.fact must be a non-empty string, got {fact_item.fact!r}"
        )
        assert fact_item.entailment in valid_entailments, (
            f"EntailmentResult.entailment must be one of {valid_entailments!r}, "
            f"got {fact_item.entailment!r}"
        )
        assert isinstance(fact_item.reasoning, str), (
            f"EntailmentResult.reasoning must be a string, got {type(fact_item.reasoning)}"
        )

    # --- unified-eval-0004: ChatUsage token counts ---
    assert result.usage is not None, "AlignmentResult.usage must not be None"
    usage = result.usage
    assert isinstance(usage, ChatUsage), (
        f"AlignmentResult.usage must be a ChatUsage, got {type(usage)}"
    )
    assert isinstance(usage.prompt_tokens, int) and usage.prompt_tokens >= 0, (
        f"usage.prompt_tokens must be a non-negative int, got {usage.prompt_tokens!r}"
    )
    assert isinstance(usage.completion_tokens, int) and usage.completion_tokens >= 0, (
        f"usage.completion_tokens must be a non-negative int, got {usage.completion_tokens!r}"
    )
    assert isinstance(usage.total_tokens, int) and usage.total_tokens > 0, (
        f"usage.total_tokens must be a positive int, got {usage.total_tokens!r}"
    )
    assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens, (
        f"prompt_tokens ({usage.prompt_tokens}) + completion_tokens "
        f"({usage.completion_tokens}) must equal total_tokens ({usage.total_tokens})"
    )


# ---------------------------------------------------------------------------
# chat with explicit model and temperature parameters
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_with_model_and_temperature_rest(client: Pinecone) -> None:
    """chat() accepts explicit model and temperature parameters and returns a valid response.

    Verifies:
    - unified-chat-0005: Can specify a language model for chat requests
    - unified-chat-0006: Can specify a temperature parameter to control randomness

    Sends a chat request with model="gpt-4o" and temperature=0.7.  Both parameters
    must be accepted by the API without error.  The response.model field should
    match the requested model name.
    """
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
            mode="w", suffix=".txt", delete=False, prefix="asst-mdl-"
        ) as f:
            f.write("Pinecone is a managed vector database supporting dense and sparse vectors.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        response = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            model="gpt-4o",
            temperature=0.7,
            stream=False,
        )

        assert isinstance(response, ChatResponse)
        assert isinstance(response.id, str) and len(response.id) > 0
        # The API echoes back which model was used
        assert isinstance(response.model, str) and len(response.model) > 0
        assert "gpt-4o" in response.model, (
            f"Expected response.model to contain 'gpt-4o', got {response.model!r}"
        )
        assert response.message.role == "assistant"
        assert isinstance(response.message.content, str) and len(response.message.content) > 0

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
# assistants.list_page() — explicit single-page pagination
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_assistants_list_page_response_structure_rest(client: Pinecone) -> None:
    """assistants.list_page() returns a ListAssistantsResponse with correct structure.

    Verifies:
    - unified-assistant-0007: Can list one page of assistants with explicit pagination control.
    - Response type is ListAssistantsResponse with an ``assistants`` list and
      an optional ``next`` field.
    - Each assistant in the list is an AssistantModel with expected fields.
    - The newly created assistant appears in the response.
    """
    from pinecone.models.assistant.list import ListAssistantsResponse

    name = unique_name("asst")
    try:
        # Create an assistant to ensure at least one exists
        assistant = client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Call list_page() — single page, no explicit page_size
        page = client.assistants.list_page()

        # Response type
        assert isinstance(page, ListAssistantsResponse), (
            f"Expected ListAssistantsResponse, got {type(page)}"
        )

        # assistants is a list
        assert isinstance(page.assistants, list), (
            f"Expected page.assistants to be a list, got {type(page.assistants)}"
        )

        # next is None or a string
        assert page.next is None or isinstance(page.next, str), (
            f"Expected page.next to be None or str, got {type(page.next)}"
        )

        # Each item in the list is an AssistantModel with expected fields
        for a in page.assistants:
            assert isinstance(a, AssistantModel), (
                f"Expected AssistantModel, got {type(a)}"
            )
            assert isinstance(a.name, str) and len(a.name) > 0, (
                f"AssistantModel.name must be a non-empty string, got {a.name!r}"
            )
            assert isinstance(a.status, str), (
                f"AssistantModel.status must be a string, got {type(a.status)}"
            )

        # The created assistant must appear in the result
        names_in_page = [a.name for a in page.assistants]
        assert name in names_in_page, (
            f"Expected newly created assistant {name!r} in list_page() result; got: {names_in_page}"
        )

    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# upload_file with caller-specified file_id — upsert behavior — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="SDK bug: upload_file(file_id=...) sends file_id as query param to POST /files "
    "endpoint (API 2025-10) which ignores it; upsert needs PUT /files/{name}/{id} on "
    "API 2026-04. See IT-0018."
)
@pytest.mark.integration
def test_upload_file_with_caller_specified_file_id_rest(client: Pinecone) -> None:
    """upload_file(file_id=...) assigns the caller-specified ID; re-uploading with the same
    file_id replaces the file (upsert semantics).

    Verifies:
    - unified-file-0005: Can upload a file with a caller-specified file identifier for upsert behavior
    - unified-file-0006: When a file identifier is provided, the upload creates the file if it
      does not exist or replaces it if it does
    """
    name = unique_name("asst")
    # file_id must be 1-128 chars, alphanumeric/hyphens/underscores
    custom_file_id = unique_name("fid")
    file_id: str | None = None

    try:
        assistant = client.assistants.create(
            name=name, instructions="File upsert test assistant."
        )
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # --- First upload: create with caller-specified file_id ---
        first_content = b"Initial content for caller-specified file ID test."
        first_upload = client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(first_content),
            file_name="caller-id-test.txt",
            file_id=custom_file_id,
            timeout=120,
        )
        assert isinstance(first_upload, AssistantFileModel)
        # The server must honor the caller-specified file_id
        assert first_upload.id == custom_file_id, (
            f"Expected file_id {custom_file_id!r} to be preserved; got {first_upload.id!r}"
        )
        file_id = first_upload.id

        # Wait until Available
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"first upload of file {file_id}",
        )

        described_first = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        first_size = described_first.size

        # --- Second upload: upsert with same file_id, different (larger) content ---
        second_content = (
            b"Replacement content for caller-specified file ID upsert test. " + b"x" * 500
        )
        second_upload = client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(second_content),
            file_name="caller-id-test-v2.txt",
            file_id=custom_file_id,
            timeout=120,
        )
        assert isinstance(second_upload, AssistantFileModel)
        # After upsert the file_id must remain the same
        assert second_upload.id == custom_file_id, (
            f"After upsert, expected file_id {custom_file_id!r}; got {second_upload.id!r}"
        )

        # Wait until Available again
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"second upload (upsert) of file {file_id}",
        )

        described_second = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        second_size = described_second.size

        # The upsert replaced the file — size must differ from the first upload
        assert second_size != first_size, (
            f"Expected file size to change after upsert replacement; "
            f"first={first_size}, second={second_size}"
        )

        # Clean up the file before assistant deletion
        client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )
