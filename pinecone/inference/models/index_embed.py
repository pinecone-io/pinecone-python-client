"""Backwards-compatibility shim — IndexEmbed model for integrated indexes.

Re-exports classes that used to live at :mod:`pinecone.inference.models.index_embed`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.

:meta private:
"""

from __future__ import annotations

import dataclasses
from typing import Any

__all__ = ["IndexEmbed"]


@dataclasses.dataclass(frozen=True)
class IndexEmbed:
    """Configuration for an integrated (model-backed) embedding index.

    Describes the embedding model and field mapping used for an integrated
    index. Legacy class preserved for backwards compatibility.

    :meta private:
    """

    model: str
    field_map: dict[str, Any]
    metric: str | None = None
    read_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    write_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the instance's field values as a plain dictionary."""
        return self.__dict__
