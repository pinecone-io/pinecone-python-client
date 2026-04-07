"""Unit tests for Index.list_paginated() and Index.list() methods."""

from __future__ import annotations

from typing import Any

import httpx
import respx

from pinecone import Index
from pinecone.models.vectors.responses import ListItem, ListResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
LIST_URL = f"{INDEX_HOST_HTTPS}/vectors/list"


def _make_list_response(
    *,
    vectors: list[dict[str, Any]] | None = None,
    pagination: dict[str, Any] | None = None,
    namespace: str = "",
) -> dict[str, object]:
    """Build a realistic list API response payload."""
    result: dict[str, object] = {
        "vectors": vectors or [],
        "namespace": namespace,
        "usage": {"readUnits": 1},
    }
    if pagination is not None:
        result["pagination"] = pagination
    return result


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# list_paginated()
# ---------------------------------------------------------------------------


class TestListPaginated:
    """Tests for list_paginated() single-page retrieval."""

    @respx.mock
    def test_list_paginated_basic(self) -> None:
        """Returns ListResponse with vector IDs."""
        respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_list_response(
                    vectors=[{"id": "v1"}, {"id": "v2"}, {"id": "v3"}],
                ),
            ),
        )
        idx = _make_index()
        result = idx.list_paginated()

        assert isinstance(result, ListResponse)
        assert len(result.vectors) == 3
        assert all(isinstance(v, ListItem) for v in result.vectors)
        assert result.vectors[0].id == "v1"
        assert result.vectors[1].id == "v2"
        assert result.vectors[2].id == "v3"

    @respx.mock
    def test_list_paginated_with_prefix(self) -> None:
        """Prefix param is forwarded as query parameter."""
        route = respx.get(LIST_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response()),
        )
        idx = _make_index()
        idx.list_paginated(prefix="doc1#")

        request = route.calls.last.request
        assert request.url.params["prefix"] == "doc1#"

    @respx.mock
    def test_list_paginated_with_limit(self) -> None:
        """Limit param is forwarded as query parameter."""
        route = respx.get(LIST_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response()),
        )
        idx = _make_index()
        idx.list_paginated(limit=50)

        request = route.calls.last.request
        assert request.url.params["limit"] == "50"

    @respx.mock
    def test_list_paginated_with_token(self) -> None:
        """Pagination token is forwarded as paginationToken query parameter."""
        route = respx.get(LIST_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response()),
        )
        idx = _make_index()
        idx.list_paginated(pagination_token="abc123")

        request = route.calls.last.request
        assert request.url.params["paginationToken"] == "abc123"

    @respx.mock
    def test_list_paginated_default_namespace(self) -> None:
        """Default namespace sends empty string as query parameter."""
        route = respx.get(LIST_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response()),
        )
        idx = _make_index()
        idx.list_paginated()

        request = route.calls.last.request
        assert request.url.params["namespace"] == ""

    @respx.mock
    def test_pagination_token_omit_when_none(self) -> None:
        """paginationToken is NOT in query params when pagination_token is None."""
        route = respx.get(LIST_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response()),
        )
        idx = _make_index()
        idx.list_paginated(namespace="ns")

        request = route.calls.last.request
        param_keys = [k for k, _ in request.url.params.multi_items()]
        assert "paginationToken" not in param_keys

    @respx.mock
    def test_pagination_token_omit_includes_when_provided(self) -> None:
        """paginationToken IS in query params with correct value when provided."""
        route = respx.get(LIST_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response()),
        )
        idx = _make_index()
        idx.list_paginated(namespace="ns", pagination_token="tok123")

        request = route.calls.last.request
        assert request.url.params["paginationToken"] == "tok123"

    @respx.mock
    def test_pagination_token_omit_last_page_no_pagination(self) -> None:
        """Response with no pagination field means current page is the last page."""
        respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_list_response(vectors=[{"id": "v1"}]),
            ),
        )
        idx = _make_index()
        result = idx.list_paginated()

        assert result.pagination is None

    @respx.mock
    def test_list_paginated_last_page_no_token(self) -> None:
        """Last page has no pagination token (unified-vec-0056)."""
        respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_list_response(vectors=[{"id": "v1"}]),
            ),
        )
        idx = _make_index()
        result = idx.list_paginated()

        assert result.pagination is None

    @respx.mock
    def test_list_paginated_returns_only_ids(self) -> None:
        """ListItem objects have id but no values/metadata (unified-vec-0029)."""
        respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_list_response(vectors=[{"id": "v1"}]),
            ),
        )
        idx = _make_index()
        result = idx.list_paginated()

        item = result.vectors[0]
        assert item.id == "v1"
        assert not hasattr(item, "values")
        assert not hasattr(item, "metadata")


# ---------------------------------------------------------------------------
# list() generator
# ---------------------------------------------------------------------------


class TestListGenerator:
    """Tests for list() auto-paginating generator."""

    @respx.mock
    def test_list_follows_pagination(self) -> None:
        """Generator follows pagination tokens across pages."""
        route = respx.get(LIST_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_make_list_response(
                        vectors=[{"id": "v1"}, {"id": "v2"}],
                        pagination={"next": "token2"},
                    ),
                ),
                httpx.Response(
                    200,
                    json=_make_list_response(vectors=[{"id": "v3"}]),
                ),
            ],
        )
        idx = _make_index()
        pages = list(idx.list())

        assert len(pages) == 2
        assert len(pages[0].vectors) == 2
        assert len(pages[1].vectors) == 1

        # First call should have no paginationToken
        first_request = route.calls[0].request
        param_keys = [k for k, _ in first_request.url.params.multi_items()]
        assert "paginationToken" not in param_keys

        # Second call should have paginationToken=token2
        second_request = route.calls[1].request
        assert second_request.url.params["paginationToken"] == "token2"

    @respx.mock
    def test_list_single_page(self) -> None:
        """Generator yields exactly one response when no pagination token."""
        respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_list_response(vectors=[{"id": "v1"}]),
            ),
        )
        idx = _make_index()
        pages = list(idx.list())

        assert len(pages) == 1

    @respx.mock
    def test_list_passes_prefix_and_limit(self) -> None:
        """Prefix and limit are forwarded to each page request."""
        route = respx.get(LIST_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_make_list_response(
                        vectors=[{"id": "v1"}],
                        pagination={"next": "tok"},
                    ),
                ),
                httpx.Response(
                    200,
                    json=_make_list_response(vectors=[{"id": "v2"}]),
                ),
            ],
        )
        idx = _make_index()
        list(idx.list(prefix="a", limit=10))

        for call in route.calls:
            assert call.request.url.params["prefix"] == "a"
            assert call.request.url.params["limit"] == "10"

    @respx.mock
    def test_list_skips_empty_pages(self) -> None:
        """Generator skips pages with empty vectors (unified-pag-0003)."""
        respx.get(LIST_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_make_list_response(
                        vectors=[{"id": "v1"}],
                        pagination={"next": "tok2"},
                    ),
                ),
                httpx.Response(
                    200,
                    json=_make_list_response(
                        vectors=[],
                        pagination={"next": "tok3"},
                    ),
                ),
                httpx.Response(
                    200,
                    json=_make_list_response(vectors=[{"id": "v2"}]),
                ),
            ],
        )
        idx = _make_index()
        pages = list(idx.list())

        assert len(pages) == 2
        assert pages[0].vectors[0].id == "v1"
        assert pages[1].vectors[0].id == "v2"

    @respx.mock
    def test_list_empty_only_pages_yields_nothing(self) -> None:
        """Generator yields nothing when all pages are empty."""
        respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_list_response(vectors=[]),
            ),
        )
        idx = _make_index()
        pages = list(idx.list())

        assert len(pages) == 0
