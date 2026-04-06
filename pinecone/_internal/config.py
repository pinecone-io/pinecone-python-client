"""Configuration for the Pinecone SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PineconeConfig:
    """SDK configuration with environment variable fallbacks.

    Args:
        api_key: Pinecone API key. Falls back to PINECONE_API_KEY env var.
        host: API host URL. Falls back to PINECONE_ENVIRONMENT env var.
        timeout: Request timeout in seconds. Defaults to 30.
        additional_headers: Extra headers to include in every request.
    """

    api_key: str = ""
    host: str = ""
    timeout: float = 30.0
    additional_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.api_key:
            env_key = os.environ.get("PINECONE_API_KEY", "")
            object.__setattr__(self, "api_key", env_key)
        if not self.host:
            env_host = os.environ.get("PINECONE_ENVIRONMENT", "")
            object.__setattr__(self, "host", env_host)
