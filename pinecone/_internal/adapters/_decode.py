"""Shared decoding helpers for adapter modules."""

from __future__ import annotations

from typing import Any, TypeVar

import msgspec

from pinecone.errors.exceptions import ResponseParsingError

T = TypeVar("T")


def decode_response(data: bytes, type: type[T]) -> T:
    """Decode *data* as JSON into an instance of *type*, wrapping decode errors.

    Raises:
        ResponseParsingError: If *data* cannot be decoded into *type*.
    """
    try:
        return msgspec.json.decode(data, type=type)
    except (msgspec.ValidationError, msgspec.DecodeError) as exc:
        raise ResponseParsingError(
            f"Failed to parse API response as {type.__name__}: {exc}",
            cause=exc,
        ) from exc


def decode_response_lax(data: bytes, type: type[T]) -> T:
    """Like :func:`decode_response` but with ``strict=False`` for lenient type coercion.

    Use when the API returns a string-encoded value for a field typed as a
    numeric type (e.g. ``record_count: "0"`` instead of ``0``).  msgspec will
    coerce the string to the declared type automatically.

    Raises:
        ResponseParsingError: If *data* cannot be decoded into *type*.
    """
    try:
        return msgspec.json.decode(data, type=type, strict=False)
    except (msgspec.ValidationError, msgspec.DecodeError) as exc:
        raise ResponseParsingError(
            f"Failed to parse API response as {type.__name__}: {exc}",
            cause=exc,
        ) from exc


def convert_response(obj: Any, type: type[T]) -> T:
    """Convert a Python object into an instance of *type*, wrapping errors.

    Raises:
        ResponseParsingError: If *obj* cannot be converted into *type*.
    """
    try:
        return msgspec.convert(obj, type)
    except (msgspec.ValidationError, msgspec.DecodeError) as exc:
        raise ResponseParsingError(
            f"Failed to parse API response as {type.__name__}: {exc}",
            cause=exc,
        ) from exc
