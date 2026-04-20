"""Backwards-compatibility shim for :mod:`pinecone.db_control.models.byoc_spec`.

Re-exports a dataclass-based ByocSpec that used to live at this path before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: The canonical pinecone.models.indexes.specs.ByocSpec is a msgspec.Struct.
# Legacy callers expect a frozen @dataclass with environment, read_capacity,
# schema fields and an asdict() method. This shim carries its own definition.

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ByocSpec:
    """Bring-your-own-cloud index deployment spec (legacy dataclass form).

    Attributes:
        environment: BYOC environment identifier (e.g. ``"aws-us-east-1-b921"``).
        read_capacity: Optional read capacity configuration.
        schema: Optional metadata schema configuration.
    """

    environment: str
    read_capacity: Any | None = None
    schema: Any | None = None

    def asdict(self) -> dict[str, Any]:
        """Return a dict with spec data nested under a ``"byoc"`` key."""
        result: dict[str, Any] = {"environment": self.environment}
        if self.read_capacity is not None:
            result["read_capacity"] = self.read_capacity
        if self.schema is not None:
            result["schema"] = self.schema
        return {"byoc": result}


__all__ = ["ByocSpec"]
