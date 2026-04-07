"""Assistants namespace — control-plane operations for Pinecone assistants."""

from __future__ import annotations

import logging
import os
import time
from typing import IO, TYPE_CHECKING, Any, Iterator, List

import msgspec
import msgspec.structs
import orjson

from pinecone._internal.adapters.assistants_adapter import AssistantsAdapter
from pinecone._internal.constants import ASSISTANT_API_VERSION
from pinecone._internal.http_client import _raise_for_status
from pinecone.errors.exceptions import (
    NotFoundError,
    PineconeError,
    PineconeTimeoutError,
    PineconeValueError,
)
from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse
from pinecone.models.assistant.context import ContextResponse
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
from pinecone.models.assistant.message import Message
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.options import ContextOptions
from pinecone.models.assistant.streaming import (
    ChatCompletionStreamChunk,
    ChatStreamChunk,
    StreamCitationChunk,
    StreamContentChunk,
    StreamMessageEnd,
    StreamMessageStart,
)
from pinecone.models.pagination import Page, Paginator

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_VALID_REGIONS = ("us", "eu")
_CREATE_POLL_INTERVAL_SECONDS = 0.5
_DELETE_POLL_INTERVAL_SECONDS = 5
_UPLOAD_POLL_INTERVAL_SECONDS = 5


class Assistants:
    """Control-plane operations for Pinecone assistants.

    Args:
        config (PineconeConfig): SDK configuration used to construct an
            HTTP client targeting the assistant API version.

    Examples:

        from pinecone import Pinecone

        pc = Pinecone(api_key="your-api-key")
        assistants = pc.assistants
    """

    def __init__(self, config: PineconeConfig) -> None:
        from pinecone._internal.http_client import HTTPClient as _HTTPClient

        self._config = config
        self._http = _HTTPClient(config, ASSISTANT_API_VERSION)
        self._adapter = AssistantsAdapter()
        self._data_plane_clients: dict[str, HTTPClient] = {}

    def close(self) -> None:
        """Close the underlying HTTP client and any cached data-plane clients."""
        self._http.close()
        for client in self._data_plane_clients.values():
            client.close()
        self._data_plane_clients.clear()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Assistants()"

    def _data_plane_http(self, assistant_name: str) -> HTTPClient:
        """Return an HTTPClient targeting the assistant's data-plane host.

        Caches clients by assistant name to avoid repeated describe calls.
        """
        if assistant_name not in self._data_plane_clients:
            from pinecone._internal.config import PineconeConfig as _PineconeConfig
            from pinecone._internal.http_client import HTTPClient as _HTTPClient

            assistant = self.describe(name=assistant_name)
            if not assistant.host:
                raise PineconeValueError(f"Assistant '{assistant_name}' has no data-plane host")
            data_config = _PineconeConfig(
                api_key=self._config.api_key,
                host=f"https://{assistant.host}",
                timeout=self._config.timeout,
                additional_headers=self._config.additional_headers,
                source_tag=self._config.source_tag or "",
                proxy_url=self._config.proxy_url or "",
                proxy_headers=self._config.proxy_headers,
                ssl_ca_certs=self._config.ssl_ca_certs,
                ssl_verify=self._config.ssl_verify,
                connection_pool_maxsize=self._config.connection_pool_maxsize,
                retry_config=self._config.retry_config,
            )
            self._data_plane_clients[assistant_name] = _HTTPClient(
                data_config, ASSISTANT_API_VERSION
            )
        return self._data_plane_clients[assistant_name]

    def upload_file(
        self,
        *,
        assistant_name: str,
        file_path: str | None = None,
        file_stream: IO[bytes] | None = None,
        file_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        multimodal: bool | None = None,
        file_id: str | None = None,
        timeout: float | None = None,
    ) -> AssistantFileModel:
        """Upload a file to a Pinecone assistant.

        Uploads a file from a local path or an in-memory byte stream, then
        polls until server-side processing completes.

        Args:
            assistant_name: Name of the target assistant.
            file_path: Path to a local file to upload. Mutually exclusive
                with *file_stream*.
            file_stream: An open byte stream to upload. Mutually exclusive
                with *file_path*. Use *file_name* to set the filename.
            file_name: Filename to associate with *file_stream*. Ignored
                when *file_path* is provided.
            metadata: Optional metadata dictionary. Sent as a JSON string.
            multimodal: Whether to enable multimodal processing for PDFs.
            file_id: Optional caller-specified file identifier for upsert
                behavior.
            timeout: Seconds to wait for processing to complete. ``None``
                (default) polls indefinitely. Use ``-1`` to return
                immediately after upload with one describe call. Raises
                :exc:`PineconeTimeoutError` if processing is not done
                before the deadline.

        Returns:
            :class:`AssistantFileModel` fetched fresh from the API after
            processing completes.

        Raises:
            :exc:`PineconeValueError`: If both or neither of *file_path*
                and *file_stream* are provided, or if *file_path* does not
                exist.
            :exc:`PineconeTimeoutError`: If processing does not complete
                before *timeout*.
            :exc:`PineconeError`: If server-side processing fails.
        """
        import json as _json

        if (file_path is None) == (file_stream is None):
            raise PineconeValueError("Exactly one of file_path or file_stream must be provided")

        opened_file: IO[bytes] | None = None
        if file_path is not None:
            if not os.path.isfile(file_path):
                raise PineconeValueError(f"File not found: {file_path}")
            opened_file = open(file_path, "rb")  # noqa: SIM115
            handle: IO[bytes] = opened_file
            upload_name = os.path.basename(file_path)
        else:
            assert file_stream is not None
            handle = file_stream
            upload_name = file_name or "upload"

        try:
            data_http = self._data_plane_http(assistant_name)

            params: dict[str, str] = {}
            if metadata is not None:
                params["metadata"] = _json.dumps(metadata)
            if multimodal is not None:
                params["multimodal"] = str(multimodal).lower()
            if file_id is not None:
                params["file_id"] = file_id

            logger.info("Uploading file %r to assistant %r", upload_name, assistant_name)
            response = data_http.post(
                f"/files/{assistant_name}",
                files={"file": (upload_name, handle)},
                params=params,
            )
            file_model = self._adapter.to_file(response.content)
            logger.debug(
                "Uploaded file %r (id=%s, status=%s)",
                upload_name,
                file_model.id,
                file_model.status,
            )
        finally:
            if opened_file is not None:
                opened_file.close()

        if timeout == -1:
            return self.describe_file(assistant_name=assistant_name, file_id=file_model.id)

        return self._poll_file_until_processed(data_http, assistant_name, file_model.id, timeout)

    def _poll_file_until_processed(
        self,
        data_http: HTTPClient,
        assistant_name: str,
        file_id: str,
        timeout: float | None,
    ) -> AssistantFileModel:
        """Poll ``GET /files/{assistant_name}/{file_id}`` until processing completes."""
        start = time.monotonic()
        while True:
            response = data_http.get(f"/files/{assistant_name}/{file_id}")
            file_model = self._adapter.to_file(response.content)

            if file_model.status != "Processing":
                if file_model.status == "ProcessingFailed":
                    error_msg = file_model.error_message or "Unknown processing error"
                    raise PineconeError(f"File processing failed for '{file_id}': {error_msg}")
                return file_model

            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(
                        f"File processing timed out after {timeout}s (operation_id={file_id})"
                    )
            time.sleep(_UPLOAD_POLL_INTERVAL_SECONDS)

    def describe_file(
        self,
        *,
        assistant_name: str,
        file_id: str,
        include_url: bool = False,
    ) -> AssistantFileModel:
        """Get the status and metadata of a file uploaded to an assistant.

        Args:
            assistant_name: Name of the assistant that owns the file.
            file_id: Unique identifier of the file to retrieve.
            include_url: If ``True``, include a signed download URL in the
                response. Defaults to ``False``.

        Returns:
            :class:`AssistantFileModel` with file metadata and status.

        Raises:
            :exc:`NotFoundError`: If the file does not exist.
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            file = pc.assistants.describe_file(
                assistant_name="my-assistant",
                file_id="file-abc123",
            )
            print(file.status)
        """
        data_http = self._data_plane_http(assistant_name)
        params: dict[str, str] = {}
        if include_url:
            params["include_url"] = "true"
        logger.info("Describing file %r in assistant %r", file_id, assistant_name)
        response = data_http.get(f"/files/{assistant_name}/{file_id}", params=params)
        return self._adapter.to_file(response.content)

    def list_files(
        self,
        *,
        assistant_name: str,
        filter: dict[str, Any] | None = None,
    ) -> list[AssistantFileModel]:
        """List all files for an assistant, automatically paginating through every page.

        Args:
            assistant_name: Name of the assistant whose files to list.
            filter: Optional metadata filter expression. Serialized to a JSON
                string before being sent to the API.

        Returns:
            A list of :class:`AssistantFileModel` objects. Returns an empty
            list when no files exist.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            files = pc.assistants.list_files(assistant_name="my-assistant")
            for f in files:
                print(f.name, f.status)
        """
        logger.info("Listing all files for assistant %r", assistant_name)
        all_files: list[AssistantFileModel] = []
        pagination_token: str | None = None

        while True:
            page = self.list_files_page(
                assistant_name=assistant_name,
                pagination_token=pagination_token,
                filter=filter,
            )
            all_files.extend(page.files)
            if page.next is None:
                break
            pagination_token = page.next

        logger.debug("Listed %d files for assistant %r", len(all_files), assistant_name)
        return all_files

    def list_files_page(
        self,
        *,
        assistant_name: str,
        page_size: int | None = None,
        pagination_token: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> ListFilesResponse:
        """List one page of files for an assistant with explicit pagination control.

        Only the parameters that are explicitly provided are sent in the
        request. Omitted parameters are not included as query params.

        Args:
            assistant_name: Name of the assistant whose files to list.
            page_size: Maximum number of files per page. Only sent when
                explicitly provided.
            pagination_token: Token from a previous response to fetch the
                next page.
            filter: Optional metadata filter expression. Serialized to a JSON
                string before being sent to the API.

        Returns:
            :class:`ListFilesResponse` with a ``files`` list and an optional
            ``next`` continuation token.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            page = pc.assistants.list_files_page(
                assistant_name="my-assistant",
                page_size=10,
            )
            for f in page.files:
                print(f.name)
            if page.next:
                next_page = pc.assistants.list_files_page(
                    assistant_name="my-assistant",
                    pagination_token=page.next,
                )
        """
        import json as _json

        data_http = self._data_plane_http(assistant_name)
        params: dict[str, str | int] = {}
        if page_size is not None:
            params["pageSize"] = page_size
        if pagination_token is not None:
            params["paginationToken"] = pagination_token
        if filter is not None:
            params["filter"] = _json.dumps(filter)

        logger.info("Listing files page for assistant %r", assistant_name)
        response = data_http.get(f"/files/{assistant_name}", params=params)
        result = self._adapter.to_file_list(response.content)
        logger.debug(
            "Listed %d files for assistant %r (has_next=%s)",
            len(result.files),
            assistant_name,
            result.next is not None,
        )
        return result

    def delete_file(
        self,
        *,
        assistant_name: str,
        file_id: str,
        timeout: float | None = None,
    ) -> None:
        """Delete a file from a Pinecone assistant.

        Sends a DELETE request, then polls every 5 seconds until the file is
        confirmed gone (404 from describe_file). Other errors during polling
        propagate immediately.

        Args:
            assistant_name: Name of the assistant that owns the file.
            file_id: Unique identifier of the file to delete.
            timeout: Seconds to wait for the file to be deleted. Use ``None``
                (default) to poll indefinitely. Use ``-1`` to return
                immediately without polling. Use a positive value to poll with
                a deadline. Raises :exc:`PineconeTimeoutError` if the file
                is not gone before the deadline.

        Returns:
            ``None``

        Raises:
            :exc:`PineconeError`: If server-side file deletion fails.
            :exc:`PineconeTimeoutError`: If the file still exists after
                *timeout* seconds.
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            pc.assistants.delete_file(
                assistant_name="my-assistant",
                file_id="file-abc123",
            )
        """
        data_http = self._data_plane_http(assistant_name)
        logger.info("Deleting file %r from assistant %r", file_id, assistant_name)
        data_http.delete(f"/files/{assistant_name}/{file_id}")
        logger.debug("Deleted file %r from assistant %r", file_id, assistant_name)

        if timeout == -1:
            return

        start = time.monotonic()
        while True:
            try:
                file_model = self.describe_file(assistant_name=assistant_name, file_id=file_id)
            except NotFoundError:
                return
            if file_model.status not in ("Deleting", None):
                error_msg = file_model.error_message or "Unknown deletion error"
                raise PineconeError(f"File deletion failed for '{file_id}': {error_msg}")
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(f"File '{file_id}' still exists after {timeout}s")
            time.sleep(_DELETE_POLL_INTERVAL_SECONDS)

    def create(
        self,
        *,
        name: str,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
        region: str = "us",
        timeout: float | None = None,
    ) -> AssistantModel:
        """Create a new Pinecone assistant.

        Creates an assistant and optionally polls until it reaches ``"Ready"``
        status. The assistant starts in ``"Initializing"`` status.

        Args:
            name (str): Name for the new assistant. Must be 1-63 characters,
                start and end with an alphanumeric character, and consist only
                of lowercase alphanumeric characters or hyphens.
            instructions (str | None): Optional directive for the assistant to
                apply to all responses. Maximum 16 KB.
            metadata (dict[str, Any] | None): Optional metadata dictionary.
                Defaults to an empty dict if not provided.
            region (str): Region to deploy the assistant in. Must be ``"us"``
                or ``"eu"`` (case-sensitive). Defaults to ``"us"``.
            timeout (float | None): Seconds to wait for the assistant to become
                ready. Use ``None`` (default) to poll indefinitely. Use ``-1``
                to return immediately without polling. Use ``0`` or a positive
                value to poll with a deadline. Raises
                :exc:`PineconeTimeoutError` if the assistant is not ready
                before the deadline.

        Returns:
            :class:`AssistantModel` describing the created assistant.

        Raises:
            :exc:`PineconeValueError`: If *region* is not ``"us"`` or ``"eu"``.
            :exc:`PineconeTimeoutError`: If the assistant does not become ready
                before the deadline.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            Create an assistant with default settings:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> assistant = pc.assistants.create(name="my-assistant")

            Create an assistant with instructions and metadata:

            >>> assistant = pc.assistants.create(
            ...     name="research-assistant",
            ...     instructions="You are a helpful research assistant.",
            ...     metadata={"team": "engineering", "version": "1"},
            ...     region="eu",
            ... )
        """
        if region not in _VALID_REGIONS:
            raise PineconeValueError(f"region must be one of {_VALID_REGIONS!r}, got {region!r}")

        body: dict[str, Any] = {
            "name": name,
            "instructions": instructions,
            "metadata": metadata if metadata is not None else {},
            "region": region,
        }

        logger.info("Creating assistant %r", name)
        response = self._http.post("/assistants", json=body)
        model = self._adapter.to_assistant(response.content)
        logger.debug("Created assistant %r (status=%s)", name, model.status)

        if timeout == -1:
            return model

        return self._poll_until_ready(name, timeout)

    def describe(self, *, name: str) -> AssistantModel:
        """Get detailed information about a named assistant.

        Args:
            name (str): The name of the assistant to describe.

        Returns:
            :class:`AssistantModel` with name, status, created_at, updated_at,
            metadata, instructions, and host.

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. 404
                when the assistant does not exist).

        Examples:

            assistant = pc.assistants.describe(name="my-assistant")
            print(assistant.status)
        """
        logger.info("Describing assistant %r", name)
        response = self._http.get(f"/assistants/{name}")
        model = self._adapter.to_assistant(response.content)
        logger.debug("Described assistant %r (status=%s)", name, model.status)
        return model

    def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[AssistantModel]:
        """List assistants in the project with transparent lazy pagination.

        Args:
            limit (int | None): Maximum number of assistants to yield across
                all pages. ``None`` (default) yields all assistants.
            pagination_token (str | None): Token to resume pagination from a
                previous call.

        Returns:
            :class:`Paginator` over :class:`AssistantModel` objects. Supports
            ``for`` loops, ``.to_list()``, ``.pages()``, and ``limit``.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            for a in pc.assistants.list():
                print(a.name, a.status)

            all_assistants = pc.assistants.list().to_list()
        """
        logger.info("Listing assistants")

        def fetch_page(token: str | None) -> Page[AssistantModel]:
            result = self.list_page(pagination_token=token)
            return Page(items=result.assistants, pagination_token=result.next)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    def list_page(
        self,
        *,
        page_size: int | None = None,
        pagination_token: str | None = None,
    ) -> ListAssistantsResponse:
        """List one page of assistants with explicit pagination control.

        Only the parameters that are explicitly provided are sent in the
        request. Omitted parameters are not included as query params.

        Args:
            page_size (int | None): Maximum number of assistants per page.
                Only sent when explicitly provided.
            pagination_token (str | None): Token from a previous response
                to fetch the next page.

        Returns:
            :class:`ListAssistantsResponse` with an ``assistants`` list and
            an optional ``next`` continuation token.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            page = pc.assistants.list_page(page_size=10)
            for a in page.assistants:
                print(a.name)
            if page.next:
                next_page = pc.assistants.list_page(pagination_token=page.next)
        """
        params: dict[str, str | int] = {}
        if page_size is not None:
            params["pageSize"] = page_size
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        logger.info("Listing assistants page")
        response = self._http.get("/assistants", params=params)
        result = self._adapter.to_assistant_list(response.content)
        logger.debug(
            "Listed %d assistants (has_next=%s)",
            len(result.assistants),
            result.next is not None,
        )
        return result

    def update(
        self,
        *,
        name: str,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AssistantModel:
        """Update an existing Pinecone assistant.

        Updates the specified assistant's instructions and/or metadata.
        Metadata is fully replaced (not merged) when provided.

        Args:
            name (str): The name of the assistant to update.
            instructions (str | None): New instructions for the assistant.
                Pass an empty string to clear existing instructions.
            metadata (dict[str, Any] | None): New metadata dictionary. Fully
                replaces any existing metadata rather than merging.

        Returns:
            :class:`AssistantModel` describing the updated assistant.

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. 404
                when the assistant does not exist).

        Examples:
            Update an assistant's instructions:

            >>> assistant = pc.assistants.update(
            ...     name="my-assistant",
            ...     instructions="You are a helpful research assistant.",
            ... )

            Replace an assistant's metadata:

            >>> assistant = pc.assistants.update(
            ...     name="my-assistant",
            ...     metadata={"team": "ml", "version": "2"},
            ... )
        """
        body: dict[str, Any] = {}
        if instructions is not None:
            body["instructions"] = instructions
        if metadata is not None:
            body["metadata"] = metadata

        logger.info("Updating assistant %r", name)
        response = self._http.patch(f"/assistants/{name}", json=body)
        model = self._adapter.to_assistant(response.content)
        logger.debug("Updated assistant %r", name)
        return model

    def delete(self, *, name: str, timeout: float | None = None) -> None:
        """Delete a Pinecone assistant by name.

        Sends a DELETE request, then polls every 5 seconds until the
        assistant is confirmed gone (404 from describe). Other errors
        during polling propagate immediately.

        Args:
            name (str): The name of the assistant to delete.
            timeout (float | None): Seconds to wait for the assistant to
                disappear. Use ``None`` (default) to poll indefinitely.
                Use ``-1`` to return immediately without polling.
                Use a positive value to poll with a deadline. Raises
                :exc:`PineconeTimeoutError` if the assistant is not gone
                before the deadline.

        Raises:
            :exc:`PineconeTimeoutError`: If the assistant still exists after
                *timeout* seconds.
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            pc.assistants.delete(name="my-assistant")

            # Return immediately without waiting for deletion
            pc.assistants.delete(name="my-assistant", timeout=-1)
        """
        logger.info("Deleting assistant %r", name)
        self._http.delete(f"/assistants/{name}")
        logger.debug("Deleted assistant %r", name)

        if timeout == -1:
            return

        start = time.monotonic()
        while True:
            try:
                self.describe(name=name)
            except NotFoundError:
                return
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(f"Assistant '{name}' still exists after {timeout}s")
            time.sleep(_DELETE_POLL_INTERVAL_SECONDS)

    def context(
        self,
        *,
        assistant_name: str,
        query: str | None = None,
        messages: List[Message | dict[str, str]] | None = None,
        filter: dict[str, Any] | None = None,
        top_k: int | None = None,
        snippet_size: int | None = None,
        multimodal: bool | None = None,
        include_binary_content: bool | None = None,
    ) -> ContextResponse:
        """Retrieve relevant context snippets from a Pinecone assistant.

        Retrieves context snippets matching a text query or conversation
        history. Exactly one of *query* or *messages* must be provided
        and non-empty.

        Args:
            assistant_name: Name of the assistant to retrieve context from.
            query: Text query to use for context retrieval. Mutually exclusive
                with *messages*. Empty string is treated as not provided.
            messages: Conversation messages to use for context retrieval.
                Mutually exclusive with *query*. Empty list is treated as not
                provided. Dicts are converted to :class:`Message` objects.
            filter: Metadata filter restricting which documents contribute
                context. Omitted from request when ``None``.
            top_k: Maximum number of context snippets to return. Omitted
                from request when ``None``.
            snippet_size: Maximum snippet size in tokens. Omitted from
                request when ``None``.
            multimodal: Whether to include image-related context snippets.
                Omitted from request when ``None``.
            include_binary_content: Whether image snippets include base64
                image data. Only meaningful when *multimodal* is ``True``.
                Omitted from request when ``None``.

        Returns:
            :class:`ContextResponse` containing the matching context snippets.

        Raises:
            :exc:`PineconeValueError`: If both or neither of *query* and
                *messages* are provided (or if they are empty).
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            Retrieve context using a text query:

            >>> response = pc.assistants.context(
            ...     assistant_name="my-assistant",
            ...     query="What is Pinecone?",
            ... )
            >>> for snippet in response.snippets:
            ...     print(snippet.content)
        """
        query_truthy = query is not None and query != ""
        messages_truthy = messages is not None and len(messages) > 0

        if query_truthy and messages_truthy:
            raise PineconeValueError("Exactly one of query or messages must be provided, not both.")
        if not query_truthy and not messages_truthy:
            raise PineconeValueError("Exactly one of query or messages must be provided.")

        body: dict[str, Any] = {}

        if query_truthy:
            body["query"] = query
        else:
            assert messages is not None
            parsed: List[Message] = [
                m if isinstance(m, Message) else Message.from_dict(m) for m in messages
            ]
            body["messages"] = [{"role": m.role, "content": m.content} for m in parsed]

        if filter is not None:
            body["filter"] = filter
        if top_k is not None:
            body["top_k"] = top_k
        if snippet_size is not None:
            body["snippet_size"] = snippet_size
        if multimodal is not None:
            body["multimodal"] = multimodal
        if include_binary_content is not None:
            body["include_binary_content"] = include_binary_content

        http = self._data_plane_http(assistant_name)
        response = http.post(f"/chat/{assistant_name}/context", json=body)
        return self._adapter.to_context_response(response.content)

    def chat(
        self,
        *,
        assistant_name: str,
        messages: List[Message | dict[str, str]],
        model: str = "gpt-4o",
        stream: bool = False,
        temperature: float | None = None,
        filter: dict[str, Any] | None = None,
        json_response: bool = False,
        include_highlights: bool = False,
        context_options: ContextOptions | dict[str, Any] | None = None,
    ) -> ChatResponse | Iterator[ChatStreamChunk]:
        """Chat with an assistant and receive citations in Pinecone-native format.

        Args:
            assistant_name (str): Name of the assistant to chat with.
            messages (list[Message | dict[str, str]]): Conversation messages.
                Dicts are converted to :class:`Message` objects; role defaults
                to ``"user"`` when not present.
            model (str): Large language model to use. Defaults to ``"gpt-4o"``.
            stream (bool): If ``True``, return a streaming iterator. Defaults
                to ``False``.
            temperature (float | None): Controls randomness. Lower values produce
                more deterministic responses. Omitted from request when ``None``.
            filter (dict[str, Any] | None): Metadata filter restricting which
                documents are used as context. Omitted from request when ``None``.
            json_response (bool): If ``True``, instruct the assistant to return
                a JSON response. Cannot be used with streaming.
            include_highlights (bool): If ``True``, include highlight snippets
                from referenced documents in citations.
            context_options (ContextOptions | dict[str, Any] | None): Options
                controlling context retrieval. Omitted from request when ``None``.

        Returns:
            :class:`ChatResponse` for non-streaming requests, or an
            :class:`Iterator[ChatStreamChunk]` for streaming requests.

        Raises:
            :exc:`PineconeValueError`: If both ``stream=True`` and
                ``json_response=True`` are specified.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            Non-streaming chat:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> response = pc.assistants.chat(
            ...     assistant_name="my-assistant",
            ...     messages=[{"content": "What is Pinecone?"}],
            ... )
        """
        if stream and json_response:
            raise PineconeValueError("json_response cannot be used with stream=True")

        parsed: List[Message] = [
            m if isinstance(m, Message) else Message.from_dict(m) for m in messages
        ]

        body: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in parsed],
            "model": model,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if filter is not None:
            body["filter"] = filter
        if json_response:
            body["json_response"] = json_response
        # Streaming requests always include include_highlights (defaults to False)
        if stream or include_highlights:
            body["include_highlights"] = include_highlights
        if context_options is not None:
            if isinstance(context_options, dict):
                body["context_options"] = context_options
            else:
                body["context_options"] = {
                    k: v
                    for k, v in msgspec.structs.asdict(context_options).items()
                    if v is not None
                }

        if stream:
            return self._chat_streaming(assistant_name=assistant_name, body=body)

        http = self._data_plane_http(assistant_name)
        response = http.post(f"/chat/{assistant_name}", json=body)
        return self._adapter.to_chat_response(response.content)

    def chat_completions(
        self,
        *,
        assistant_name: str,
        messages: List[Message | dict[str, str]],
        model: str = "gpt-4o",
        stream: bool = False,
        temperature: float | None = None,
        filter: dict[str, Any] | None = None,
    ) -> ChatCompletionResponse | Iterator[ChatCompletionStreamChunk]:
        """Chat with an assistant using an OpenAI-compatible interface.

        Returns responses in OpenAI chat completion format. Useful when you
        need inline citations or OpenAI-compatible responses. Has limited
        functionality compared to the standard :meth:`chat` interface — does
        not support ``include_highlights``, ``context_options``, or
        ``json_response`` parameters.

        The model parameter accepts any string value and is not validated
        client-side. Known models include ``"gpt-4o"``, ``"gpt-4.1"``,
        ``"o4-mini"``, ``"claude-3-5-sonnet"``, ``"claude-3-7-sonnet"``,
        and ``"gemini-2.5-pro"``.

        Args:
            assistant_name (str): Name of the assistant to chat with.
            messages (list[Message | dict[str, str]]): Conversation messages.
                Dicts are converted to :class:`Message` objects; role defaults
                to ``"user"`` when not present.
            model (str): Large language model to use. Defaults to ``"gpt-4o"``.
                Not validated client-side — any string is accepted.
            stream (bool): If ``True``, return a streaming iterator. Defaults
                to ``False``.
            temperature (float | None): Controls randomness. Lower values produce
                more deterministic responses. Omitted from request when ``None``.
            filter (dict[str, Any] | None): Metadata filter restricting which
                documents are used as context. Omitted from request when ``None``.

        Returns:
            :class:`ChatCompletionResponse` for non-streaming requests, or an
            :class:`Iterator[ChatCompletionStreamChunk]` for streaming requests.

        Raises:
            :exc:`ApiError`: If the API returns an error response.
        """
        parsed: List[Message] = [
            m if isinstance(m, Message) else Message.from_dict(m) for m in messages
        ]

        body: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in parsed],
            "model": model,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if filter is not None:
            body["filter"] = filter

        if stream:
            return self._chat_completions_streaming(assistant_name=assistant_name, body=body)

        http = self._data_plane_http(assistant_name)
        response = http.post(f"/chat/{assistant_name}/chat/completions", json=body)
        return self._adapter.to_chat_completion_response(response.content)

    def _chat_streaming(
        self,
        *,
        assistant_name: str,
        body: dict[str, Any],
    ) -> Iterator[ChatStreamChunk]:
        """Stream Pinecone-native chat chunks via SSE.

        POSTs to ``/chat/{assistant_name}`` with ``stream=True`` in the body,
        parses each SSE line, and yields typed chunk objects dispatched by the
        ``type`` field.

        Args:
            assistant_name: Name of the assistant to chat with.
            body: Pre-built request body (must include ``stream=True``).

        Yields:
            :class:`StreamMessageStart`, :class:`StreamContentChunk`,
            :class:`StreamCitationChunk`, or :class:`StreamMessageEnd`
            depending on the ``type`` field of each SSE chunk.

        Raises:
            :exc:`ApiError`: If the server returns an HTTP error.
        """
        http = self._data_plane_http(assistant_name)
        with http._client.stream(
            "POST",
            f"/chat/{assistant_name}",
            content=orjson.dumps(body),
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        ) as response:
            if not response.is_success:
                response.read()
            _raise_for_status(response)
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].lstrip()
                if not line:
                    continue
                chunk_data: dict[str, Any] = orjson.loads(line)
                chunk_type = chunk_data.get("type", "")
                if chunk_type == "message_start":
                    yield msgspec.convert(chunk_data, StreamMessageStart)
                elif chunk_type == "content_chunk":
                    yield msgspec.convert(chunk_data, StreamContentChunk)
                elif chunk_type == "citation":
                    yield msgspec.convert(chunk_data, StreamCitationChunk)
                elif chunk_type == "message_end":
                    yield msgspec.convert(chunk_data, StreamMessageEnd)

    def _chat_completions_streaming(
        self,
        *,
        assistant_name: str,
        body: dict[str, Any],
    ) -> Iterator[ChatCompletionStreamChunk]:
        """Stream OpenAI-compatible chat completion chunks via SSE.

        POSTs to ``/chat/{assistant_name}/chat/completions`` with ``stream=True``
        in the body and yields each SSE line parsed as a
        :class:`ChatCompletionStreamChunk`.

        Args:
            assistant_name: Name of the assistant to chat with.
            body: Pre-built request body (must include ``stream=True``).

        Yields:
            :class:`ChatCompletionStreamChunk` for each non-empty SSE line.

        Raises:
            :exc:`ApiError`: If the server returns an HTTP error.
        """
        http = self._data_plane_http(assistant_name)
        with http._client.stream(
            "POST",
            f"/chat/{assistant_name}/chat/completions",
            content=orjson.dumps(body),
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        ) as response:
            if not response.is_success:
                response.read()
            _raise_for_status(response)
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].lstrip()
                if not line:
                    continue
                yield msgspec.convert(orjson.loads(line), ChatCompletionStreamChunk)

    def _poll_until_ready(self, name: str, timeout: float | None) -> AssistantModel:
        """Poll ``GET /assistants/{name}`` until status is ``"Ready"`` or timeout."""
        start = time.monotonic()
        while True:
            response = self._http.get(f"/assistants/{name}")
            model = self._adapter.to_assistant(response.content)
            if model.status == "Ready":
                return model
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(
                        f"Assistant '{name}' not ready after {timeout}s. "
                        f"Check status with pc.assistants.describe(name='{name}')."
                    )
            time.sleep(_CREATE_POLL_INTERVAL_SECONDS)
