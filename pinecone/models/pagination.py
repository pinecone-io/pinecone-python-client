"""Pagination types for lazy iteration over paginated API results."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Generic, TypeVar

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

    Examples:
        Iterate over results one item at a time:

        .. code-block:: python

            paginator = pc.assistants.list()
            for assistant in paginator:
                print(assistant.name)

        Collect all results into a list:

        .. code-block:: python

            all_assistants = pc.assistants.list().to_list()
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
        """Iterate over pages rather than individual items.

        When ``limit`` is set, yields full pages until the remaining budget is
        exhausted, then yields a truncated final page and stops.

        Returns:
            :class:`~collections.abc.Generator` yielding :class:`Page` objects.
            Each page has an ``items`` list and an optional ``pagination_token``.

        Examples:
            Process results page by page:

            .. code-block:: python

                for page in pc.assistants.list().pages():
                    for assistant in page.items:
                        print(assistant.name)
        """
        count = 0
        token: str | None = self._initial_token
        while True:
            page = self._fetch_page(token)
            self._pagination_token = page.pagination_token
            if self._limit is not None:
                remaining = self._limit - count
                if remaining <= 0:
                    return
                if len(page.items) > remaining:
                    yield Page(items=page.items[:remaining], pagination_token=None)
                    return
                count += len(page.items)
            yield page
            if page.pagination_token is None:
                return
            token = page.pagination_token

    def to_list(self) -> list[T]:
        """Fetch all items across all pages into a list.

        Returns:
            list of all items.

        Examples:
            Collect all assistants at once:

            .. code-block:: python

                all_assistants = pc.assistants.list().to_list()
        """
        return list(self)

    def __repr__(self) -> str:
        has_more = self._pagination_token is not None
        parts = [f"has_more={has_more!r}"]
        if self._limit is not None:
            parts.append(f"limit={self._limit!r}")
        return f"Paginator({', '.join(parts)})"


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

    Examples:
        Iterate over results one item at a time:

        .. code-block:: python

            paginator = async_pc.assistants.list()
            async for assistant in paginator:
                print(assistant.name)

        Collect all results into a list:

        .. code-block:: python

            paginator = async_pc.assistants.list()
            all_assistants = await paginator.to_list()
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
        """Iterate over pages rather than individual items.

        When ``limit`` is set, yields full pages until the remaining budget is
        exhausted, then yields a truncated final page and stops.

        Returns:
            :class:`~collections.abc.AsyncGenerator` yielding :class:`Page`
            objects. Each page has an ``items`` list and an optional
            ``pagination_token``.

        Examples:
            Process results page by page:

                async for page in async_pc.assistants.list().pages():
                    for assistant in page.items:
                        print(assistant.name)
        """
        count = 0
        token: str | None = self._initial_token
        while True:
            page = await self._fetch_page(token)
            self._pagination_token = page.pagination_token
            if self._limit is not None:
                remaining = self._limit - count
                if remaining <= 0:
                    return
                if len(page.items) > remaining:
                    yield Page(items=page.items[:remaining], pagination_token=None)
                    return
                count += len(page.items)
            yield page
            if page.pagination_token is None:
                return
            token = page.pagination_token

    async def to_list(self) -> list[T]:
        """Fetch all items across all pages into a list.

        Returns:
            list of all items.

        Examples:
            Collect all assistants at once:

                paginator = async_pc.assistants.list()
                all_assistants = await paginator.to_list()
        """
        return [item async for item in self]

    def __repr__(self) -> str:
        has_more = self._pagination_token is not None
        parts = [f"has_more={has_more!r}"]
        if self._limit is not None:
            parts.append(f"limit={self._limit!r}")
        return f"AsyncPaginator({', '.join(parts)})"
