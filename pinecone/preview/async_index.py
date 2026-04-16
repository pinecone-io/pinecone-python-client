"""Async preview index data-plane wrapper (2026-01.alpha)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pinecone.preview.async_documents import AsyncPreviewDocuments as AsyncPreviewDocuments

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.http_client import AsyncHTTPClient

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
        http: Async HTTP client from the parent :class:`~pinecone.AsyncPinecone` instance.
        config: SDK configuration shared with the parent client.
        host: Normalized data-plane host URL. Provide either ``host`` or ``_host_provider``.
        _host_provider: Async callable that resolves the host on first data-plane use.
            Used internally when the factory is called with ``name=``; callers should
            not pass this directly.
    """

    def __init__(
        self,
        http: AsyncHTTPClient,
        config: PineconeConfig,
        host: str | None = None,
        _host_provider: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        self._resolved_host: str | None = host
        self._host_provider = _host_provider
        self._http = http
        self._config = config
        self._documents: AsyncPreviewDocuments | None = (
            AsyncPreviewDocuments(http=http, config=config, host=host) if host is not None else None
        )

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
        if self._resolved_host is None:
            raise RuntimeError(
                "Host not yet resolved; call _resolve_host() or a data-plane method first."
            )
        return self._resolved_host

    @property
    def documents(self) -> AsyncPreviewDocuments:
        """Documents sub-namespace for data-plane operations on this index."""
        if self._documents is None:
            raise RuntimeError(
                "Host not yet resolved; call _resolve_host() or a data-plane method first."
            )
        return self._documents

    async def _resolve_host(self) -> str:
        """Resolve and cache the data-plane host, invoking the host provider at most once."""
        if self._resolved_host is None:
            if self._host_provider is None:
                raise RuntimeError("AsyncPreviewIndex: no host or host_provider configured.")
            self._resolved_host = await self._host_provider()
            self._documents = AsyncPreviewDocuments(
                http=self._http, config=self._config, host=self._resolved_host
            )
        return self._resolved_host

    def __repr__(self) -> str:
        return f"AsyncPreviewIndex(host={self._resolved_host!r})"
