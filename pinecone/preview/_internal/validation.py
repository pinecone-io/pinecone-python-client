"""Preview-specific input validation helpers."""

from __future__ import annotations

import re

from pinecone.errors.exceptions import PineconeValueError

_TAG_KEY_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_TAGS = 20
_MAX_KEY_LEN = 80
_MAX_VAL_LEN = 120


def validate_tags(tags: dict[str, str] | None) -> None:
    """Raise PineconeValueError if any tag violates backend constraints."""
    if tags is None:
        return
    if len(tags) > _MAX_TAGS:
        raise PineconeValueError(f"Tags exceeded the maximum of {_MAX_TAGS}. Got {len(tags)} tags.")
    for key, value in tags.items():
        if len(key) > _MAX_KEY_LEN:
            raise PineconeValueError(f"Tag key {key!r} exceeds the {_MAX_KEY_LEN}-character limit.")
        if not _TAG_KEY_RE.match(key):
            raise PineconeValueError(
                f"Tag key {key!r} has invalid characters. Must be alphanumeric or '_', '-'."
            )
        if len(value) > _MAX_VAL_LEN:
            raise PineconeValueError(
                f"Tag value for key {key!r} exceeds the {_MAX_VAL_LEN}-character limit."
            )
        if not value.isascii() or not value.isprintable():
            raise PineconeValueError(
                f"Tag value for key {key!r} contains invalid characters. "
                "Only printable ASCII characters are allowed."
            )
