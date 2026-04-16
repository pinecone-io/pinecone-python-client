"""Preview namespace — pre-release API features not covered by SemVer.

Access via ``pc.preview``. See docs/conventions/preview-channel.md for
the full lifecycle (introduction, iteration, graduation, retirement).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pinecone.preview.schema_builder import PreviewSchemaBuilder as PreviewSchemaBuilder

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
    from pinecone.preview.async_indexes import AsyncPreviewIndexes
    from pinecone.preview.indexes import PreviewIndexes

__all__ = ["AsyncPreview", "Preview", "PreviewSchemaBuilder"]


class Preview:
    """Sync preview namespace — routes to per-area preview classes.

    .. admonition:: Preview
       :class: warning

       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Args:
        http: Shared HTTP client from the parent :class:`~pinecone.Pinecone` instance.
        config: SDK configuration shared with the parent client.
    """

    def __init__(self, http: HTTPClient, config: PineconeConfig) -> None:
        self._http = http
        self._config = config
        self._indexes: PreviewIndexes | None = None

    @property
    def indexes(self) -> PreviewIndexes:
        """Access the preview indexes control-plane namespace.

        Lazily instantiated on first access. Reuses the parent client's
        configuration and credentials.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Returns:
            :class:`~pinecone.preview.indexes.PreviewIndexes` instance.
        """
        if self._indexes is None:
            from pinecone.preview.indexes import PreviewIndexes

            self._indexes = PreviewIndexes(config=self._config)
        return self._indexes

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
        http: Shared async HTTP client from the parent :class:`~pinecone.AsyncPinecone` instance.
        config: SDK configuration shared with the parent client.
    """

    def __init__(self, http: AsyncHTTPClient, config: PineconeConfig) -> None:
        self._http = http
        self._config = config
        self._indexes: AsyncPreviewIndexes | None = None

    @property
    def indexes(self) -> AsyncPreviewIndexes:
        """Access the async preview indexes control-plane namespace.

        Lazily instantiated on first access. Reuses the parent client's
        configuration and credentials.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Returns:
            :class:`~pinecone.preview.async_indexes.AsyncPreviewIndexes` instance.
        """
        if self._indexes is None:
            from pinecone.preview.async_indexes import AsyncPreviewIndexes

            self._indexes = AsyncPreviewIndexes(config=self._config)
        return self._indexes

    def __repr__(self) -> str:
        return "AsyncPreview()"
