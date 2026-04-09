"""Shared helpers for sync and async data plane clients."""

from __future__ import annotations

from typing import Any

from pinecone._internal.config import normalize_host
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.vector import Vector


def _validate_host(host: str) -> str:
    """Validate and normalize an index host URL.

    Raises:
        ValidationError: If the host is empty or does not look like a real hostname.
    """
    if not host or not host.strip():
        raise ValidationError("host must be a non-empty string")
    normalized = normalize_host(host.strip())
    # Strip scheme for the dot/localhost check
    bare = normalized
    for prefix in ("https://", "http://"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    if "." not in bare and "localhost" not in bare.lower():
        raise ValidationError(
            f"host {host!r} does not appear to be a valid URL (must contain a dot or 'localhost')"
        )
    return normalized


def _vector_to_dict(v: Vector) -> dict[str, Any]:
    """Serialize a Vector to a dict matching the API wire format."""
    sv = v.sparse_values
    if sv is not None:
        sv_dict: dict[str, Any] = {"indices": sv.indices, "values": sv.values}
        md = v.metadata
        if md is not None:
            return {"id": v.id, "values": v.values, "sparseValues": sv_dict, "metadata": md}
        return {"id": v.id, "values": v.values, "sparseValues": sv_dict}
    md = v.metadata
    if md is not None:
        return {"id": v.id, "values": v.values, "metadata": md}
    return {"id": v.id, "values": v.values}
