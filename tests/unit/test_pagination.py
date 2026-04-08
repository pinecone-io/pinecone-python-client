"""Unit tests for Page, Paginator, and AsyncPaginator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from pinecone.models.pagination import AsyncPaginator, Page, Paginator

# ---------------------------------------------------------------------------
# Page tests
# ---------------------------------------------------------------------------


def test_page_has_more_with_token() -> None:
    page: Page[int] = Page(items=[1], pagination_token="tok")
    assert page.has_more is True


def test_page_has_more_without_token() -> None:
    page: Page[int] = Page(items=[1], pagination_token=None)
    assert page.has_more is False


def test_page_repr() -> None:
    page: Page[int] = Page(items=[1, 2], pagination_token="tok")
    r = repr(page)
    assert "items" in r
    assert "pagination_token" in r
    assert "[1, 2]" in r
    assert "'tok'" in r


# ---------------------------------------------------------------------------
# Paginator (sync) tests
# ---------------------------------------------------------------------------


def test_paginator_single_page() -> None:
    """Single page with no token yields all items."""
    fetch = MagicMock(return_value=Page(items=[1, 2, 3], pagination_token=None))
    pag: Paginator[int] = Paginator(fetch_page=fetch)
    assert list(pag) == [1, 2, 3]
    fetch.assert_called_once_with(None)


def test_paginator_multi_page() -> None:
    """Two pages yield items from both."""
    fetch = MagicMock(
        side_effect=[
            Page(items=[1, 2], pagination_token="p2"),
            Page(items=[3, 4], pagination_token=None),
        ]
    )
    pag: Paginator[int] = Paginator(fetch_page=fetch)
    assert list(pag) == [1, 2, 3, 4]


def test_paginator_with_limit() -> None:
    """5 items across 2 pages, limit=3 yields 3 items."""
    fetch = MagicMock(
        side_effect=[
            Page(items=[1, 2, 3], pagination_token="p2"),
            Page(items=[4, 5], pagination_token=None),
        ]
    )
    pag: Paginator[int] = Paginator(fetch_page=fetch, limit=3)
    assert list(pag) == [1, 2, 3]


def test_paginator_limit_zero() -> None:
    """limit=0 yields nothing."""
    fetch = MagicMock(return_value=Page(items=[1, 2, 3], pagination_token=None))
    pag: Paginator[int] = Paginator(fetch_page=fetch, limit=0)
    assert list(pag) == []


def test_paginator_empty_first_page() -> None:
    """First page empty with no token yields nothing."""
    fetch = MagicMock(return_value=Page(items=[], pagination_token=None))
    pag: Paginator[int] = Paginator(fetch_page=fetch)
    assert list(pag) == []


def test_paginator_pages_method() -> None:
    """.pages() yields Page objects."""
    p1: Page[int] = Page(items=[1, 2], pagination_token="p2")
    p2: Page[int] = Page(items=[3], pagination_token=None)
    fetch = MagicMock(side_effect=[p1, p2])
    pag: Paginator[int] = Paginator(fetch_page=fetch)
    pages = list(pag.pages())
    assert pages == [p1, p2]
    assert all(isinstance(p, Page) for p in pages)


def test_paginator_to_list() -> None:
    """.to_list() returns a list of all items."""
    fetch = MagicMock(
        side_effect=[
            Page(items=[1, 2], pagination_token="p2"),
            Page(items=[3], pagination_token=None),
        ]
    )
    pag: Paginator[int] = Paginator(fetch_page=fetch)
    assert pag.to_list() == [1, 2, 3]


def test_paginator_pagination_token_updated() -> None:
    """After full iteration, .pagination_token is None."""
    fetch = MagicMock(
        side_effect=[
            Page(items=[1], pagination_token="p2"),
            Page(items=[2], pagination_token=None),
        ]
    )
    pag: Paginator[int] = Paginator(fetch_page=fetch)
    list(pag)
    assert pag.pagination_token is None


def test_paginator_repr() -> None:
    fetch = MagicMock(return_value=Page(items=[], pagination_token=None))
    pag: Paginator[int] = Paginator(fetch_page=fetch)
    assert repr(pag) == "Paginator()"


def test_paginator_pages_with_limit() -> None:
    """3 pages of 3 items each, limit=5 → first page full, second truncated to 2, third never fetched."""
    fetch = MagicMock(
        side_effect=[
            Page(items=[1, 2, 3], pagination_token="p2"),
            Page(items=[4, 5, 6], pagination_token="p3"),
            Page(items=[7, 8, 9], pagination_token=None),
        ]
    )
    pag: Paginator[int] = Paginator(fetch_page=fetch, limit=5)
    result = list(pag.pages())
    assert len(result) == 2
    assert result[0].items == [1, 2, 3]
    assert result[1].items == [4, 5]
    assert fetch.call_count == 2


def test_paginator_pages_without_limit() -> None:
    """Without limit, all pages are yielded in full."""
    fetch = MagicMock(
        side_effect=[
            Page(items=[1, 2], pagination_token="p2"),
            Page(items=[3, 4], pagination_token="p3"),
            Page(items=[5], pagination_token=None),
        ]
    )
    pag: Paginator[int] = Paginator(fetch_page=fetch)
    result = list(pag.pages())
    assert len(result) == 3
    assert result[0].items == [1, 2]
    assert result[1].items == [3, 4]
    assert result[2].items == [5]


def test_paginator_pages_limit_exact_boundary() -> None:
    """limit equals total items → all pages yielded without truncation."""
    fetch = MagicMock(
        side_effect=[
            Page(items=[1, 2, 3], pagination_token="p2"),
            Page(items=[4, 5, 6], pagination_token=None),
        ]
    )
    pag: Paginator[int] = Paginator(fetch_page=fetch, limit=6)
    result = list(pag.pages())
    assert len(result) == 2
    assert result[0].items == [1, 2, 3]
    assert result[1].items == [4, 5, 6]


# ---------------------------------------------------------------------------
# AsyncPaginator tests
# ---------------------------------------------------------------------------


async def test_async_paginator_single_page() -> None:
    """Single page with no token yields all items."""
    fetch = AsyncMock(return_value=Page(items=[1, 2, 3], pagination_token=None))
    pag: AsyncPaginator[int] = AsyncPaginator(fetch_page=fetch)
    result = [item async for item in pag]
    assert result == [1, 2, 3]
    fetch.assert_called_once_with(None)


async def test_async_paginator_multi_page() -> None:
    """Two pages yield items from both."""
    fetch = AsyncMock(
        side_effect=[
            Page(items=[1, 2], pagination_token="p2"),
            Page(items=[3, 4], pagination_token=None),
        ]
    )
    pag: AsyncPaginator[int] = AsyncPaginator(fetch_page=fetch)
    result = [item async for item in pag]
    assert result == [1, 2, 3, 4]


async def test_async_paginator_with_limit() -> None:
    """5 items across 2 pages, limit=3 yields 3 items."""
    fetch = AsyncMock(
        side_effect=[
            Page(items=[1, 2, 3], pagination_token="p2"),
            Page(items=[4, 5], pagination_token=None),
        ]
    )
    pag: AsyncPaginator[int] = AsyncPaginator(fetch_page=fetch, limit=3)
    result = [item async for item in pag]
    assert result == [1, 2, 3]


async def test_async_paginator_pages_method() -> None:
    """.pages() yields Page objects."""
    p1: Page[int] = Page(items=[1, 2], pagination_token="p2")
    p2: Page[int] = Page(items=[3], pagination_token=None)
    fetch = AsyncMock(side_effect=[p1, p2])
    pag: AsyncPaginator[int] = AsyncPaginator(fetch_page=fetch)
    pages = [p async for p in pag.pages()]
    assert pages == [p1, p2]
    assert all(isinstance(p, Page) for p in pages)


async def test_async_paginator_to_list() -> None:
    """.to_list() returns a list of all items."""
    fetch = AsyncMock(
        side_effect=[
            Page(items=[1, 2], pagination_token="p2"),
            Page(items=[3], pagination_token=None),
        ]
    )
    pag: AsyncPaginator[int] = AsyncPaginator(fetch_page=fetch)
    result = await pag.to_list()
    assert result == [1, 2, 3]


def test_async_paginator_repr() -> None:
    fetch = AsyncMock(return_value=Page(items=[], pagination_token=None))
    pag: AsyncPaginator[int] = AsyncPaginator(fetch_page=fetch)
    assert repr(pag) == "AsyncPaginator()"


async def test_async_paginator_pages_with_limit() -> None:
    """3 pages of 3 items each, limit=5 → first page full, second truncated to 2, third never fetched."""
    fetch = AsyncMock(
        side_effect=[
            Page(items=[1, 2, 3], pagination_token="p2"),
            Page(items=[4, 5, 6], pagination_token="p3"),
            Page(items=[7, 8, 9], pagination_token=None),
        ]
    )
    pag: AsyncPaginator[int] = AsyncPaginator(fetch_page=fetch, limit=5)
    result = [p async for p in pag.pages()]
    assert len(result) == 2
    assert result[0].items == [1, 2, 3]
    assert result[1].items == [4, 5]
    assert fetch.call_count == 2
