"""Configuration for the Pinecone SDK."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def normalize_host(host: str | None) -> str:
    """Normalize a host string by ensuring it has an https:// prefix.

    - If host is None or empty, return "".
    - If host doesn't start with http:// or https://, prepend https://.
    - Existing http:// or https:// prefixes are preserved as-is.
    """
    if not host:
        return ""
    if not host.startswith(("http://", "https://")):
        return f"https://{host}"
    return host


def normalize_source_tag(tag: str | None) -> str:
    """Normalize a source tag string.

    - Lowercase the input.
    - Strip characters not in [a-z0-9_ :].
    - Replace spaces with underscores.
    """
    if not tag:
        return ""
    lowered = tag.lower()
    cleaned = re.sub(r"[^a-z0-9_ :]", "", lowered)
    return cleaned.replace(" ", "_")


def _parse_additional_headers_env() -> dict[str, str]:
    """Parse PINECONE_ADDITIONAL_HEADERS env var as JSON."""
    raw = os.environ.get("PINECONE_ADDITIONAL_HEADERS", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        logger.warning(
            "Failed to parse PINECONE_ADDITIONAL_HEADERS env var, ignoring: %s",
            raw,
        )
    return {}


@dataclass(frozen=True)
class PineconeConfig:
    """SDK configuration with environment variable fallbacks.

    Args:
        api_key: Pinecone API key. Falls back to PINECONE_API_KEY env var.
        host: API host URL. Falls back to PINECONE_CONTROLLER_HOST env var.
        timeout: Request timeout in seconds. Defaults to 30.
        additional_headers: Extra headers to include in every request.
        source_tag: Source tag for User-Agent string.
        proxy_url: HTTP proxy URL.
        ssl_ca_certs: Path to CA certificate bundle.
        ssl_verify: Whether to verify SSL certificates.
    """

    api_key: str = ""
    host: str = ""
    timeout: float = 30.0
    additional_headers: dict[str, str] = field(default_factory=dict)
    source_tag: str = ""
    proxy_url: str = ""
    ssl_ca_certs: str | None = None
    ssl_verify: bool = True
    connection_pool_maxsize: int = 0

    def __repr__(self) -> str:
        masked = f"...{self.api_key[-4:]}" if len(self.api_key) >= 4 else "***"
        return (
            f"PineconeConfig("
            f"api_key='{masked}', "
            f"host='{self.host}', "
            f"timeout={self.timeout}, "
            f"additional_headers={self.additional_headers!r}, "
            f"source_tag='{self.source_tag}', "
            f"proxy_url='{self.proxy_url}', "
            f"ssl_ca_certs={self.ssl_ca_certs!r}, "
            f"ssl_verify={self.ssl_verify}, "
            f"connection_pool_maxsize={self.connection_pool_maxsize}"
            f")"
        )

    def __post_init__(self) -> None:
        if not self.api_key:
            env_key = os.environ.get("PINECONE_API_KEY", "")
            object.__setattr__(self, "api_key", env_key)
        if not self.host:
            env_host = os.environ.get("PINECONE_CONTROLLER_HOST", "")
            object.__setattr__(self, "host", normalize_host(env_host))
        else:
            object.__setattr__(self, "host", normalize_host(self.host))
        if not self.additional_headers:
            env_headers = _parse_additional_headers_env()
            if env_headers:
                object.__setattr__(self, "additional_headers", env_headers)
        if self.source_tag:
            object.__setattr__(self, "source_tag", normalize_source_tag(self.source_tag))
