"""Input validation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

from pinecone.errors.exceptions import ValidationError


@overload
def require_non_empty(name: str, value: str) -> None: ...


@overload
def require_non_empty(name: str, value: list[Any]) -> None: ...


def require_non_empty(name: str, value: str | list[Any]) -> None:
    """Raise ValidationError if value is empty, whitespace-only, or an empty list."""
    if isinstance(value, list):
        if not value:
            raise ValidationError(f"{name} must be a non-empty list")
    else:
        if not value or not value.strip():
            raise ValidationError(f"{name} must be a non-empty string")


def require_positive(name: str, value: int) -> None:
    """Raise ValidationError if value is not a positive integer."""
    if value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {value}")


def require_one_of(name: str, value: str, allowed: Sequence[str]) -> None:
    """Raise ValidationError if *value* is not in the *allowed* set."""
    if value not in allowed:
        opts = ", ".join(repr(a) for a in allowed)
        raise ValidationError(f"{name} must be one of {opts}, got {value!r}")
