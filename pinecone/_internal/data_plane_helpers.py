"""Shared helpers for sync and async data plane clients."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pinecone._internal.config import normalize_host
from pinecone.errors.exceptions import PineconeValueError, ValidationError
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


_ALLOWED_SEARCH_VECTOR_KEYS: frozenset[str] = frozenset(
    {"values", "sparse_indices", "sparse_values"}
)


def _normalize_search_vector_dict(vector: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a sparse/hybrid search vector dict.

    Raises:
        PineconeValueError: when *vector* contains a key outside the supported set
            (``values``, ``sparse_indices``, ``sparse_values``).
    """
    unknown = set(vector.keys()) - _ALLOWED_SEARCH_VECTOR_KEYS
    if unknown:
        raise PineconeValueError(
            f"Unsupported keys in search vector dict: {sorted(unknown)}. "
            f"Allowed keys are {sorted(_ALLOWED_SEARCH_VECTOR_KEYS)}."
        )
    result: dict[str, Any] = {}
    if "values" in vector:
        result["values"] = list(vector["values"])
    if "sparse_indices" in vector:
        result["sparse_indices"] = list(vector["sparse_indices"])
    if "sparse_values" in vector:
        result["sparse_values"] = list(vector["sparse_values"])
    return result


def _vector_to_dict(v: Vector) -> dict[str, Any]:
    """Serialize a Vector to a dict matching the API wire format."""
    id_ = v.id
    vals = v.values
    if v.sparse_values is not None:
        sv = v.sparse_values
        sv_dict: dict[str, Any] = {"indices": sv.indices, "values": sv.values}
        if v.metadata is not None:
            return {"id": id_, "values": vals, "sparseValues": sv_dict, "metadata": v.metadata}
        return {"id": id_, "values": vals, "sparseValues": sv_dict}
    if v.metadata is not None:
        return {"id": id_, "values": vals, "metadata": v.metadata}
    return {"id": id_, "values": vals}
