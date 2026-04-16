"""Backwards-compatibility shim for :mod:`pinecone.config.pinecone_config`.

Re-exports classes that used to live at :mod:`pinecone.config.pinecone_config`
before the ``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: no canonical equivalent for the static-builder interface;
# this shim is the only definition
from __future__ import annotations

import json
import logging
import os
from typing import Any

from pinecone.config.config import Config

__all__ = ["PineconeConfig"]

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "https://api.pinecone.io"

_VALID_CONFIG_FIELDS = frozenset(Config._fields)


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


class PineconeConfig:
    """Legacy static builder for Pinecone SDK configuration.

    Returns a :class:`~pinecone.config.config.Config` named tuple.
    """

    @staticmethod
    def build(
        api_key: str = "",
        host: str = "",
        additional_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Config:
        """Build a :class:`~pinecone.config.config.Config` from explicit args and env vars.

        Args:
            api_key: Pinecone API key.
            host: API host URL. Falls back to ``PINECONE_CONTROLLER_HOST`` env var,
                then to ``https://api.pinecone.io``.
            additional_headers: Extra headers for every request. Falls back to
                ``PINECONE_ADDITIONAL_HEADERS`` env var (parsed as JSON).
            **kwargs: Additional fields forwarded to :class:`Config` if they match
                known field names.

        Returns:
            A populated :class:`Config` named tuple.
        """
        if not host:
            host = os.environ.get("PINECONE_CONTROLLER_HOST", _DEFAULT_HOST)
        if additional_headers is None:
            additional_headers = _parse_additional_headers_env()
        extra = {k: v for k, v in kwargs.items() if k in _VALID_CONFIG_FIELDS}
        return Config(
            api_key=api_key,
            host=host,
            additional_headers=additional_headers,
            **extra,
        )
