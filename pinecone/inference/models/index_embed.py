"""The :class:`IndexEmbed` model for legacy integrated indexes.

This module is the class's only definition. ``from pinecone import
IndexEmbed`` and ``from pinecone.inference.models import IndexEmbed`` both
reach it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

__all__ = ["IndexEmbed"]


@dataclasses.dataclass(frozen=True)
class IndexEmbed:
    """Which model embeds a legacy integrated index, and which field it embeds.

    Accepted as the ``embed`` argument of
    :meth:`~pinecone.client.indexes.Indexes.create_for_model`, alongside a plain
    dict and :class:`~pinecone.models.indexes.specs.EmbedConfig`, which is the
    shape new code should prefer.

    >>> from pinecone import IndexEmbed
    >>> embed = IndexEmbed(model="llama-text-embed-v2", field_map={"text": "chunk_text"})
    >>> embed.as_dict()["model"]
    'llama-text-embed-v2'
    """

    model: str
    field_map: dict[str, Any]
    metric: str | None = None
    read_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    write_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the instance's field values as a plain dictionary."""
        return self.__dict__
