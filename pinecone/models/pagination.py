"""Pagination types for lazy iteration over paginated API results."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator
from typing import Awaitable, Generic, TypeVar

T = TypeVar("T")


class Page(Generic[T]):
    """A single page of results from a paginated API."""

    def __init__(
        self,
        *,
        items: list[T],
        pagination_token: str | None,
    ) -> None:
        self.items = items
        self.pagination_token = pagination_token

    @property
    def has_more(self) -> bool:
        """True if more pages are available."""
        return self.pagination_token is not None

    def __repr__(self) -> str:
        return f"Page(items={self.items!r}, pagination_token={self.pagination_token!r})"


class Paginator(Generic[T]):
    """Lazy iterator over paginated API results (sync).

    Fetches pages on demand. Supports item-level iteration, page-level access
    via :meth:`pages`, bulk collection via :meth:`to_list`, and resumption via
    the :attr:`pagination_token` property.

    Args:
        fetch_page: Callable that takes an optional pagination token and returns
            a :class:`Page`.
        initial_token: Token to start pagination from. ``None`` starts from
            the beginning.
        limit: Maximum number of items to yield across all pages. ``None``
            yields all items.
    """

    def __init__(
        self,
        *,
        fetch_page: Callable[[str | None], Page[T]],
        initial_token: str | None = None,
        limit: int | None = None,
    ) -> None:
        self._fetch_page = fetch_page
        self._initial_token = initial_token
        self._limit = limit
        self._pagination_token: str | None = initial_token

    @property
    def pagination_token(self) -> str | None:
        """Token for the next page, or ``None`` if all pages have been fetched."""
        return self._pagination_token

    def __iter__(self) -> Generator[T, None, None]:
        count = 0
        token: str | None = self._initial_token
        while True:
            page = self._fetch_page(token)
            self._pagination_token = page.pagination_token
            for item in page.items:
                if self._limit is not None and count >= self._limit:
                    return
                yield item
                count += 1
            if page.pagination_token is None:
                return
            token = page.pagination_token

    def pages(self) -> Generator[Page[T], None, None]:
        """Iterate over pages rather than individual items."""
        token: str | None = self._initial_token
        while True:
            page = self._fetch_page(token)
            self._pagination_token = page.pagination_token
            yield page
            if page.pagination_token is None:
                return
            token = page.pagination_token

    def to_list(self) -> list[T]:
        """Fetch all items into a list."""
        return list(self)

    def __repr__(self) -> str:
        return "Paginator()"


class AsyncPaginator(Generic[T]):
    """Async lazy iterator over paginated API results.

    Fetches pages on demand. Supports item-level async iteration, page-level
    access via :meth:`pages`, bulk collection via :meth:`to_list`, and
    resumption via the :attr:`pagination_token` property.

    Args:
        fetch_page: Async callable that takes an optional pagination token and
            returns a :class:`Page`.
        initial_token: Token to start pagination from. ``None`` starts from
            the beginning.
        limit: Maximum number of items to yield across all pages. ``None``
            yields all items.
    """

    def __init__(
        self,
        *,
        fetch_page: Callable[[str | None], Awaitable[Page[T]]],
        initial_token: str | None = None,
        limit: int | None = None,
    ) -> None:
        self._fetch_page = fetch_page
        self._initial_token = initial_token
        self._limit = limit
        self._pagination_token: str | None = initial_token

    @property
    def pagination_token(self) -> str | None:
        """Token for the next page, or ``None`` if all pages have been fetched."""
        return self._pagination_token

    async def __aiter__(self) -> AsyncGenerator[T, None]:
        count = 0
        token: str | None = self._initial_token
        while True:
            page = await self._fetch_page(token)
            self._pagination_token = page.pagination_token
            for item in page.items:
                if self._limit is not None and count >= self._limit:
                    return
                yield item
                count += 1
            if page.pagination_token is None:
                return
            token = page.pagination_token

    async def pages(self) -> AsyncGenerator[Page[T], None]:
        """Iterate over pages rather than individual items."""
        token: str | None = self._initial_token
        while True:
            page = await self._fetch_page(token)
            self._pagination_token = page.pagination_token
            yield page
            if page.pagination_token is None:
                return
            token = page.pagination_token

    async def to_list(self) -> list[T]:
        """Fetch all items into a list."""
        return [item async for item in self]

    def __repr__(self) -> str:
        return "AsyncPaginator()"
