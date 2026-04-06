"""Index and IndexStatus response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class IndexStatus(Struct, kw_only=True):
    """Status of an index.

    Attributes:
        ready: Whether the index is ready to accept requests.
        state: Current state of the index (e.g. ``"Ready"``, ``"Initializing"``).
    """

    ready: bool
    state: str


class IndexModel(Struct, kw_only=True):
    """Response model for a Pinecone index.

    Attributes:
        name: The name of the index.
        metric: Distance metric used for similarity search (e.g. ``"cosine"``,
            ``"euclidean"``, ``"dotproduct"``).
        host: The hostname where this index is served.
        status: Current status of the index.
        spec: Deployment specification as a dict containing either ``"serverless"``,
            ``"pod"``, or ``"byoc"`` configuration.
        vector_type: Type of vectors stored (default: ``"dense"``).
        dimension: Dimensionality of vectors in the index, or ``None`` for
            indexes that infer dimension from the first upsert.
        deletion_protection: Whether deletion protection is enabled
            (``"enabled"`` or ``"disabled"``).
        tags: User-defined key-value tags attached to the index, or ``None``
            if no tags are set.
    """

    name: str
    metric: str
    host: str
    status: IndexStatus
    spec: dict[str, Any]
    vector_type: str = "dense"
    dimension: int | None = None
    deletion_protection: str = "disabled"
    tags: dict[str, str] | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. index['name'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None
