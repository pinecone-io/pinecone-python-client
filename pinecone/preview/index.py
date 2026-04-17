"""Preview index data-plane wrapper (2026-01.alpha)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pinecone.preview.documents import PreviewDocuments as PreviewDocuments

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig

__all__ = ["PreviewDocuments", "PreviewIndex"]


class PreviewIndex:
    """Data-plane wrapper for a preview index.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Obtain via ``pc.preview.index(name=...)`` or ``pc.preview.index(host=...)``.

    Args:
        host: Normalized data-plane host URL for this index.
        config: SDK configuration shared with the parent client.
    """

    def __init__(self, host: str, config: PineconeConfig) -> None:
        self._host = host
        self._config = config
        self.documents = PreviewDocuments(config=config, host=host)

    @property
    def host(self) -> str:
        """Data-plane host URL for this index.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.
        """
        return self._host

    def close(self) -> None:
        """Close the underlying HTTP client. Idempotent.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.
        """
        self.documents.close()

    def __enter__(self) -> PreviewIndex:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"PreviewIndex(host={self._host!r})"
