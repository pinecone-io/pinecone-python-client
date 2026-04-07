"""Assistant response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class AssistantModel(Struct, kw_only=True):
    """Response model for a Pinecone assistant.

    Attributes:
        name: The name of the assistant.
        status: Current status of the assistant (e.g. ``"Initializing"``,
            ``"Ready"``, ``"Terminating"``, ``"Failed"``,
            ``"InitializationFailed"``).
        created_at: ISO 8601 timestamp when the assistant was created.
        updated_at: ISO 8601 timestamp when the assistant was last updated.
        metadata: Optional metadata dictionary associated with the assistant,
            or ``None`` if not set.
        instructions: Optional description or directive for the assistant
            to apply to all responses, or ``None`` if not set.
        host: The host where the assistant is deployed, or ``None`` if
            not yet available.
    """

    name: str
    status: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] | None = None
    instructions: str | None = None
    host: str | None = None
