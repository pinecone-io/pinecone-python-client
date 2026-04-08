"""Async Assistants namespace — control-plane operations for Pinecone assistants."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.assistants_adapter import AssistantsAdapter
from pinecone._internal.constants import ASSISTANT_API_VERSION, ASSISTANT_EVALUATION_BASE_URL
from pinecone.errors.exceptions import PineconeTimeoutError, PineconeValueError
from pinecone.models.assistant.list import ListAssistantsResponse
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.pagination import AsyncPaginator, Page

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)

_VALID_REGIONS = ("us", "eu")
_CREATE_POLL_INTERVAL_SECONDS = 0.5


class AsyncAssistants:
    """Async control-plane operations for Pinecone assistants.

    Args:
        config (PineconeConfig): SDK configuration used to construct an
            HTTP client targeting the assistant API version.

    Examples:

        from pinecone import AsyncPinecone

        async with AsyncPinecone(api_key="your-api-key") as pc:
            assistants = pc.assistants
    """

    def __init__(self, config: PineconeConfig) -> None:
        from pinecone._internal.config import PineconeConfig as _PineconeConfig
        from pinecone._internal.http_client import AsyncHTTPClient as _AsyncHTTPClient

        self._config = config
        self._http = _AsyncHTTPClient(config, ASSISTANT_API_VERSION)
        self._adapter = AssistantsAdapter()
        self._data_plane_clients: dict[str, AsyncHTTPClient] = {}

        eval_config = _PineconeConfig(
            api_key=config.api_key,
            host=ASSISTANT_EVALUATION_BASE_URL,
            timeout=config.timeout,
            additional_headers=config.additional_headers,
            source_tag=config.source_tag or "",
            proxy_url=config.proxy_url or "",
            proxy_headers=config.proxy_headers,
            ssl_ca_certs=config.ssl_ca_certs,
            ssl_verify=config.ssl_verify,
            connection_pool_maxsize=config.connection_pool_maxsize,
            retry_config=config.retry_config,
        )
        self._eval_http = _AsyncHTTPClient(eval_config, ASSISTANT_API_VERSION)

    async def close(self) -> None:
        """Close the underlying HTTP client and any cached data-plane clients."""
        await self._http.close()
        await self._eval_http.close()
        for client in self._data_plane_clients.values():
            await client.close()
        self._data_plane_clients.clear()

    async def _data_plane_http(self, assistant_name: str) -> AsyncHTTPClient:
        """Return an AsyncHTTPClient targeting the assistant's data-plane host.

        Caches clients by assistant name to avoid repeated describe calls.
        """
        if assistant_name not in self._data_plane_clients:
            from pinecone._internal.config import PineconeConfig as _PineconeConfig
            from pinecone._internal.http_client import AsyncHTTPClient as _AsyncHTTPClient

            assistant = await self.describe(name=assistant_name)
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
            self._data_plane_clients[assistant_name] = _AsyncHTTPClient(
                data_config, ASSISTANT_API_VERSION
            )
        return self._data_plane_clients[assistant_name]

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "AsyncAssistants()"

    async def create(
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
            instructions (str | None): Optional directive for the assistant.
                Maximum 16 KB.
            metadata (dict[str, Any] | None): Optional metadata dictionary.
                Defaults to an empty dict if not provided.
            region (str): Region to deploy the assistant in. Must be ``"us"``
                or ``"eu"`` (case-sensitive). Defaults to ``"us"``.
            timeout (float | None): Seconds to wait for the assistant to become
                ready. Use ``None`` (default) to poll indefinitely. Use ``-1``
                to return immediately without polling. Use ``0`` or a positive
                value to poll with a deadline.

        Returns:
            :class:`AssistantModel` describing the created assistant.

        Raises:
            :exc:`PineconeValueError`: If *region* is not ``"us"`` or ``"eu"``.
            :exc:`PineconeTimeoutError`: If the assistant does not become ready
                before the deadline.
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            >>> from pinecone import AsyncPinecone
            >>> async with AsyncPinecone(api_key="your-api-key") as pc:
            ...     assistant = await pc.assistants.create(name="my-assistant")
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
        response = await self._http.post("/assistants", json=body)
        model = self._adapter.to_assistant(response.content)
        logger.debug("Created assistant %r (status=%s)", name, model.status)

        if timeout == -1:
            return model

        return await self._poll_until_ready(name, timeout)

    async def describe(self, *, name: str) -> AssistantModel:
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

            assistant = await pc.assistants.describe(name="my-assistant")
            print(assistant.status)
        """
        logger.info("Describing assistant %r", name)
        response = await self._http.get(f"/assistants/{name}")
        model = self._adapter.to_assistant(response.content)
        logger.debug("Described assistant %r (status=%s)", name, model.status)
        return model

    def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> AsyncPaginator[AssistantModel]:
        """List assistants in the project with transparent lazy pagination.

        Args:
            limit (int | None): Maximum number of assistants to yield across
                all pages. ``None`` (default) yields all assistants.
            pagination_token (str | None): Token to resume pagination from a
                previous call.

        Returns:
            :class:`AsyncPaginator` over :class:`AssistantModel` objects.
            Supports ``async for`` loops, ``.to_list()``, ``.pages()``, and
            ``limit``.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            async for a in pc.assistants.list():
                print(a.name, a.status)

            all_assistants = await pc.assistants.list().to_list()
        """
        logger.info("Listing assistants")

        async def fetch_page(token: str | None) -> Page[AssistantModel]:
            result = await self.list_page(pagination_token=token)
            return Page(items=result.assistants, pagination_token=result.next)

        return AsyncPaginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    async def list_page(
        self,
        *,
        page_size: int | None = None,
        pagination_token: str | None = None,
    ) -> ListAssistantsResponse:
        """List one page of assistants with explicit pagination control.

        Args:
            page_size (int | None): Maximum number of assistants per page.
            pagination_token (str | None): Token from a previous response
                to fetch the next page.

        Returns:
            :class:`ListAssistantsResponse` with an ``assistants`` list and
            an optional ``next`` continuation token.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            page = await pc.assistants.list_page(page_size=10)
            for a in page.assistants:
                print(a.name)
            if page.next:
                next_page = await pc.assistants.list_page(pagination_token=page.next)
        """
        params: dict[str, str | int] = {}
        if page_size is not None:
            params["pageSize"] = page_size
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        logger.info("Listing assistants page")
        response = await self._http.get("/assistants", params=params)
        result = self._adapter.to_assistant_list(response.content)
        logger.debug(
            "Listed %d assistants (has_next=%s)",
            len(result.assistants),
            result.next is not None,
        )
        return result

    async def update(
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

            >>> assistant = await pc.assistants.update(
            ...     name="my-assistant",
            ...     instructions="You are a helpful research assistant.",
            ... )

            Replace an assistant's metadata:

            >>> assistant = await pc.assistants.update(
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
        response = await self._http.patch(f"/assistants/{name}", json=body)
        model = self._adapter.to_assistant(response.content)
        logger.debug("Updated assistant %r", name)
        return model

    async def delete(self, *, name: str) -> None:
        """Delete a Pinecone assistant by name.

        Args:
            name (str): The name of the assistant to delete.

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. 404
                when the assistant does not exist).

        Examples:

            await pc.assistants.delete(name="my-assistant")
        """
        logger.info("Deleting assistant %r", name)
        await self._http.delete(f"/assistants/{name}")
        logger.debug("Deleted assistant %r", name)

    async def _poll_until_ready(self, name: str, timeout: float | None) -> AssistantModel:
        """Poll ``GET /assistants/{name}`` until status is ``"Ready"`` or timeout."""
        start = time.monotonic()
        while True:
            response = await self._http.get(f"/assistants/{name}")
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
            await asyncio.sleep(_CREATE_POLL_INTERVAL_SECONDS)
