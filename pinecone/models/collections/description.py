"""CollectionDescription NamedTuple for basic collection metadata."""

from __future__ import annotations

from typing import NamedTuple


class CollectionDescription(NamedTuple):
    """Basic metadata describing a collection.

    Attributes:
        name: The name of the collection.
        source: The source index used to create the collection.
    """

    name: str
    source: str
