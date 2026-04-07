"""Assistants namespace — control-plane operations for Pinecone assistants."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.assistants_adapter import AssistantsAdapter
from pinecone._internal.constants import ASSISTANT_API_VERSION
from pinecone.errors.exceptions import PineconeTimeoutError, PineconeValueError
from pinecone.models.assistant.list import ListAssistantsResponse
from pinecone.models.assistant.model import AssistantModel

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig

logger = logging.getLogger(__name__)

_VALID_REGIONS = ("us", "eu")
_CREATE_POLL_INTERVAL_SECONDS = 0.5


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
        from pinecone._internal.http_client import HTTPClient

        self._config = config
        self._http = HTTPClient(config, ASSISTANT_API_VERSION)
        self._adapter = AssistantsAdapter()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Assistants()"

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

    def list(self) -> list[AssistantModel]:
        """List all assistants in the project.

        Automatically paginates through all pages, collecting every
        assistant into a single list.

        Returns:
            A list of :class:`AssistantModel` objects. Returns an empty
            list when no assistants exist.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            assistants = pc.assistants.list()
            for a in assistants:
                print(a.name, a.status)
        """
        logger.info("Listing all assistants")
        all_assistants: list[AssistantModel] = []
        pagination_token: str | None = None

        while True:
            page = self.list_page(pagination_token=pagination_token)
            all_assistants.extend(page.assistants)
            if page.next is None:
                break
            pagination_token = page.next

        logger.debug("Listed %d assistants", len(all_assistants))
        return all_assistants

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

    def delete(self, *, name: str) -> None:
        """Delete a Pinecone assistant by name.

        Args:
            name (str): The name of the assistant to delete.

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. 404
                when the assistant does not exist).

        Examples:

            pc.assistants.delete(name="my-assistant")
        """
        logger.info("Deleting assistant %r", name)
        self._http.delete(f"/assistants/{name}")
        logger.debug("Deleted assistant %r", name)

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
