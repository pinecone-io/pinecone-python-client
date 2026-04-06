"""Usage information model."""

from __future__ import annotations

from msgspec import Struct


class Usage(Struct, rename="camel", kw_only=True):
    """Read/write unit usage information included in responses.

    Attributes:
        read_units: Number of read units consumed, or ``None`` if the
            operation did not consume read units.
        write_units: Number of write units consumed, or ``None`` if the
            operation did not consume write units.
    """

    read_units: int | None = None
    write_units: int | None = None
