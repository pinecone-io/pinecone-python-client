"""Assistants namespace — control-plane operations for Pinecone assistants."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pinecone._internal.constants import ASSISTANT_API_VERSION

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig


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

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Assistants()"
