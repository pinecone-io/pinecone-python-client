"""Usage information model."""

from __future__ import annotations

from msgspec import Struct


class Usage(Struct, kw_only=True):
    """Read/write unit usage information included in responses."""

    read_units: int | None = None
    write_units: int | None = None
