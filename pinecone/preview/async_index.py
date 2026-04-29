"""Async preview index data-plane wrapper (2026-01.alpha)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pinecone.preview.async_documents import AsyncPreviewDocuments as AsyncPreviewDocuments

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pinecone._internal.config import PineconeConfig

__all__ = ["AsyncPreviewDocuments", "AsyncPreviewIndex"]


class AsyncPreviewIndex:
    """Async data-plane wrapper for a preview index.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Obtain via ``pc.preview.index(name=...)`` or ``pc.preview.index(host=...)``.

    Args:
        config: SDK configuration shared with the parent client.
        host: Normalized data-plane host URL. Provide either ``host`` or ``_host_provider``.
        _host_provider: Async callable that resolves the host on first data-plane use.
            Used internally when the factory is called with ``name=``; callers should
            not pass this directly.

    Examples:
        >>> import asyncio
        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> async def main() -> None:
        ...     async with pc.preview.index(name="articles-en-preview") as index:
        ...         response = await index.documents.upsert(
        ...             namespace="articles-en",
        ...             documents=[{"_id": "doc-1", "title": "Introduction to vectors"}],
        ...         )
        >>> asyncio.run(main())

        Explicit open/close when a context manager is not convenient:

        >>> async def main() -> None:
        ...     index = pc.preview.index(host="https://my-index.svc.pinecone.io")
        ...     try:
        ...         response = await index.documents.upsert(
        ...             namespace="articles-en",
        ...             documents=[{"_id": "doc-1", "title": "Introduction to vectors"}],
        ...         )
        ...     finally:
        ...         await index.close()
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        config: PineconeConfig,
        host: str | None = None,
        _host_provider: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        self._resolved_host: str | None = host
        self._host_provider = _host_provider
        self._config = config
        if host is not None:
            self._documents: AsyncPreviewDocuments = AsyncPreviewDocuments(config=config, host=host)
        else:
            self._documents = AsyncPreviewDocuments(config=config, _host_provider=_host_provider)

    @property
    def host(self) -> str:
        """Data-plane host URL for this index.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Raises:
            :exc:`RuntimeError`: If the host has not yet been resolved. When the index
                was created with ``name=`` rather than ``host=``, the host is resolved
                lazily on the first data-plane call. Access ``host`` only after at least
                one data-plane operation has completed.

        Examples:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.preview.index(host="https://my-index.svc.pinecone.io")
            >>> print(index.host)
            https://my-index.svc.pinecone.io
        """
        if self._resolved_host is None:
            raise RuntimeError(
                "Host not yet resolved; call _resolve_host() or a data-plane method first."
            )
        return self._resolved_host

    @property
    def documents(self) -> AsyncPreviewDocuments:
        """Documents sub-namespace for data-plane operations on this index.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Examples:

            >>> import asyncio
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> async def main() -> None:
            ...     async with pc.preview.index(name="articles-en-preview") as index:
            ...         docs = index.documents
            ...         response = await docs.upsert(
            ...             namespace="articles-en",
            ...             documents=[{"_id": "doc-1", "title": "Introduction to vectors"}],
            ...         )
            >>> asyncio.run(main())
        """
        return self._documents

    async def _resolve_host(self) -> str:
        """Resolve and cache the data-plane host, invoking the host provider at most once."""
        if self._resolved_host is None:
            if self._host_provider is None:
                raise RuntimeError("AsyncPreviewIndex: no host or host_provider configured.")
            self._resolved_host = await self._host_provider()
        return self._resolved_host

    async def close(self) -> None:
        """Close the underlying HTTP client if initialized. Idempotent.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Examples:

            >>> import asyncio
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> async def main() -> None:
            ...     index = pc.preview.index(host="https://my-index.svc.pinecone.io")
            ...     await index.close()
            >>> asyncio.run(main())
        """
        await self._documents.close()

    async def __aenter__(self) -> AsyncPreviewIndex:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"AsyncPreviewIndex(host={self._resolved_host!r})"
