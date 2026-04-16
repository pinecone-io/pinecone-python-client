"""Preview index status response models (2026-01.alpha API)."""

from __future__ import annotations

from msgspec import Struct

__all__ = ["PreviewIndexStatus"]


class PreviewIndexStatus(Struct, kw_only=True):
    """Index status information.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        ready: Whether the index is ready to serve queries.
        state: Current operational state. Observed values: ``"Initializing"``,
            ``"Ready"``, ``"ScalingUp"``, ``"ScalingDown"``, ``"Terminating"``.
    """

    ready: bool
    state: str
