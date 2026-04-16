"""Preview namespace — pre-release API features not covered by SemVer.

Access via ``pc.preview``. See docs/conventions/preview-channel.md for
the full lifecycle (introduction, iteration, graduation, retirement).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig

__all__ = ["AsyncPreview", "Preview"]


class Preview:
    """Sync preview namespace — routes to per-area preview classes.

    .. admonition:: Preview
       :class: warning

       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Args:
        config: SDK configuration used to construct per-area HTTP clients.
    """

    def __init__(self, config: PineconeConfig) -> None:
        self._config = config

    def __repr__(self) -> str:
        return "Preview()"


class AsyncPreview:
    """Async preview namespace — routes to per-area async preview classes.

    .. admonition:: Preview
       :class: warning

       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Args:
        config: SDK configuration used to construct per-area HTTP clients.
    """

    def __init__(self, config: PineconeConfig) -> None:
        self._config = config

    def __repr__(self) -> str:
        return "AsyncPreview()"
