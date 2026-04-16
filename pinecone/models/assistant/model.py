"""Assistant response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone._internal.config import normalize_host
from pinecone.models.assistant._legacy_methods import AssistantModelLegacyMethodsMixin
from pinecone.models.assistant._mixin import StructDictMixin


class AssistantModel(
    AssistantModelLegacyMethodsMixin,
    StructDictMixin,
    Struct,
    dict=True,
    kw_only=True,
):
    """Response model for a Pinecone assistant.

    Attributes:
        name: The name of the assistant.
        status: Current status of the assistant (e.g. ``"Initializing"``,
            ``"Ready"``, ``"Terminating"``, ``"Failed"``,
            ``"InitializationFailed"``).
        created_at: ISO 8601 timestamp when the assistant was created, or
            ``None`` if not returned by the API.
        updated_at: ISO 8601 timestamp when the assistant was last updated, or
            ``None`` if not returned by the API.
        metadata: Optional metadata dictionary associated with the assistant,
            or ``None`` if not set.
        instructions: Optional description or directive for the assistant
            to apply to all responses, or ``None`` if not set.
        host: The host where the assistant is deployed, or ``None`` if
            not yet available.
    """

    name: str
    status: str
    metadata: dict[str, Any] | None = None
    instructions: str | None = None
    host: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize host to always include https:// scheme when present."""
        if self.host is not None:
            self.host = normalize_host(self.host)
