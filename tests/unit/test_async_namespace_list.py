"""Unit tests for AsyncIndex.list_namespaces and list_namespaces_paginated."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone import AsyncIndex
from pinecone.models.namespaces.models import ListNamespacesResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
NS_URL = f"{INDEX_HOST_HTTPS}/namespaces"


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# list_namespaces_paginated
# ---------------------------------------------------------------------------


class TestAsyncListNamespacesPaginated:
    """Async list_namespaces_paginated (unified-ns-0005, unified-ns-0006)."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_list_namespaces_paginated_basic(self) -> None:
        """Return a page with 2 namespaces and no pagination token."""
        respx.get(NS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": [
                        {"name": "ns1", "record_count": 10},
                        {"name": "ns2", "record_count": 20},
                    ],
                    "pagination": None,
                    "total_count": 2,
                },
            ),
        )
        idx = _make_async_index()
        result = await idx.list_namespaces_paginated()

        assert isinstance(result, ListNamespacesResponse)
        assert len(result.namespaces) == 2
        assert result.namespaces[0].name == "ns1"
        assert result.namespaces[0].record_count == 10
        assert result.namespaces[1].name == "ns2"
        assert result.namespaces[1].record_count == 20
        assert result.pagination is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_list_namespaces_paginated_with_params(self) -> None:
        """Verify prefix, limit, and paginationToken are sent as query params."""
        route = respx.get(NS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": [{"name": "prod-ns1", "record_count": 5}],
                    "pagination": None,
                    "total_count": 1,
                },
            ),
        )
        idx = _make_async_index()
        await idx.list_namespaces_paginated(
            prefix="prod-",
            limit=50,
            pagination_token="tok123",
        )

        assert route.called
        request = respx.calls.last.request
        assert request.url.params["prefix"] == "prod-"
        assert request.url.params["limit"] == "50"
        assert request.url.params["paginationToken"] == "tok123"


# ---------------------------------------------------------------------------
# list_namespaces (auto-pagination generator)
# ---------------------------------------------------------------------------


class TestAsyncListNamespaces:
    """Async list_namespaces auto-pagination (unified-ns-0007, unified-ns-0008)."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_list_namespaces_auto_pagination(self) -> None:
        """Two pages: first has pagination token, second has None."""
        call_count = 0

        def _side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    200,
                    json={
                        "namespaces": [{"name": "ns1", "record_count": 10}],
                        "pagination": {"next": "page2-token"},
                        "total_count": 2,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "namespaces": [{"name": "ns2", "record_count": 20}],
                    "pagination": None,
                    "total_count": 2,
                },
            )

        respx.get(NS_URL).mock(side_effect=_side_effect)

        idx = _make_async_index()
        pages: list[ListNamespacesResponse] = []
        async for page in idx.list_namespaces():
            pages.append(page)

        assert len(pages) == 2
        assert pages[0].namespaces[0].name == "ns1"
        assert pages[1].namespaces[0].name == "ns2"
        assert pages[1].pagination is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_list_namespaces_single_page(self) -> None:
        """Generator yields one page when no pagination token in response."""
        respx.get(NS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": [
                        {"name": "ns1", "record_count": 5},
                        {"name": "ns2", "record_count": 15},
                    ],
                    "pagination": None,
                    "total_count": 2,
                },
            ),
        )

        idx = _make_async_index()
        pages: list[ListNamespacesResponse] = []
        async for page in idx.list_namespaces():
            pages.append(page)

        assert len(pages) == 1
        assert len(pages[0].namespaces) == 2
