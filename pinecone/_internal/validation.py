"""Input validation utilities."""

from __future__ import annotations

from pinecone.errors.exceptions import ValidationError


def require_non_empty(name: str, value: str) -> None:
    """Raise ValidationError if value is empty or whitespace-only."""
    if not value or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")


def require_positive(name: str, value: int) -> None:
    """Raise ValidationError if value is not a positive integer."""
    if value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {value}")
