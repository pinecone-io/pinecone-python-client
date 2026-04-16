"""Async preview index data-plane wrapper (2026-01.alpha)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.http_client import AsyncHTTPClient

__all__ = ["AsyncPreviewIndex"]


class AsyncPreviewDocuments:
    """Async documents sub-namespace for a preview index.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Placeholder — full implementation lands in PS-0012.
    """

    def __getattr__(self, name: str) -> object:
        raise NotImplementedError(
            f"AsyncPreviewDocuments.{name} is not yet implemented. "
            "Document operations will be available in a future release."
        )


class AsyncPreviewIndex:
    """Async data-plane wrapper for a preview index.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Obtain via ``await pc.preview.index(name=...)`` or ``pc.preview.index(host=...)``.

    Args:
        host: Normalized data-plane host URL for this index.
        http: Async HTTP client from the parent :class:`~pinecone.AsyncPinecone` instance.
        config: SDK configuration shared with the parent client.
    """

    def __init__(self, host: str, http: AsyncHTTPClient, config: PineconeConfig) -> None:
        self._host = host
        self._http = http
        self._config = config
        self.documents = AsyncPreviewDocuments()

    @property
    def host(self) -> str:
        """Data-plane host URL for this index."""
        return self._host

    def __repr__(self) -> str:
        return f"AsyncPreviewIndex(host={self._host!r})"
