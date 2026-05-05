"""Unit tests for AsyncIndex bulk import operations."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from pinecone import AsyncIndex
from pinecone.errors.exceptions import ValidationError
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
IMPORTS_URL = f"{INDEX_HOST_HTTPS}/bulk/imports"


def _make_import_response(
    *,
    id: str = "import-123",
    uri: str = "s3://my-bucket/data/",
    status: str = "Pending",
    created_at: str = "2025-01-01T00:00:00Z",
    finished_at: str | None = None,
    percent_complete: float | None = None,
    records_imported: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a realistic import API response payload."""
    result: dict[str, Any] = {
        "id": id,
        "uri": uri,
        "status": status,
        "createdAt": created_at,
    }
    if finished_at is not None:
        result["finishedAt"] = finished_at
    if percent_complete is not None:
        result["percentComplete"] = percent_complete
    if records_imported is not None:
        result["recordsImported"] = records_imported
    if error is not None:
        result["error"] = error
    return result


def _make_list_response(
    imports: list[dict[str, Any]],
    *,
    pagination_token: str | None = None,
) -> dict[str, Any]:
    """Build a realistic list-imports API response payload."""
    result: dict[str, Any] = {"data": imports}
    if pagination_token is not None:
        result["pagination"] = {"next": pagination_token}
    return result


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# start_import
# ---------------------------------------------------------------------------


class TestAsyncStartImport:
    """Tests for AsyncIndex.start_import()."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_start_import(self) -> None:
        """Mock POST, verify body omits errorMode when error_mode is not supplied."""
        route = respx.post(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-001"}),
        )
        idx = _make_async_index()
        result = await idx.start_import(uri="s3://bucket/data/")

        assert isinstance(result, StartImportResponse)
        assert result.id == "import-001"

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["uri"] == "s3://bucket/data/"
        assert "errorMode" not in body

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_start_import_error_mode_default(self) -> None:
        """Verify errorMode absent by default and present when explicitly supplied."""
        route_none = respx.post(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-none"}),
        )
        idx = _make_async_index()
        await idx.start_import(uri="s3://bucket/data/")
        body_none = json.loads(route_none.calls.last.request.content)
        assert "errorMode" not in body_none

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_start_import_error_mode_abort_explicit(self) -> None:
        """Verify errorMode present and correct when error_mode='abort'."""
        route = respx.post(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-abort"}),
        )
        idx = _make_async_index()
        await idx.start_import(uri="s3://bucket/data/", error_mode="abort")
        body = json.loads(route.calls.last.request.content)
        assert body["errorMode"] == {"onError": "abort"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_start_import_error_mode_continue_explicit(self) -> None:
        """Verify errorMode present and correct when error_mode='continue'."""
        route = respx.post(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-continue"}),
        )
        idx = _make_async_index()
        await idx.start_import(uri="s3://bucket/data/", error_mode="continue")
        body = json.loads(route.calls.last.request.content)
        assert body["errorMode"] == {"onError": "continue"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_start_import_abort_mode(self) -> None:
        """Verify error_mode='ABORT' normalizes to 'abort' in body."""
        route = respx.post(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-002"}),
        )
        idx = _make_async_index()
        await idx.start_import(uri="s3://bucket/data/", error_mode="ABORT")

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["errorMode"] == {"onError": "abort"}

    @pytest.mark.asyncio
    async def test_async_start_import_error_mode_validation(self) -> None:
        """Verify invalid error_mode raises ValidationError."""
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="error_mode"):
            await idx.start_import(uri="s3://bucket/data/", error_mode="skip")

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_start_import_with_integration_id(self) -> None:
        """Verify integrationId appears in body when set."""
        route = respx.post(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-003"}),
        )
        idx = _make_async_index()
        await idx.start_import(uri="s3://bucket/data/", integration_id="int-456")

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["integrationId"] == "int-456"


# ---------------------------------------------------------------------------
# describe_import
# ---------------------------------------------------------------------------


class TestAsyncDescribeImport:
    """Tests for AsyncIndex.describe_import()."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_describe_import(self) -> None:
        """Mock GET, verify returns ImportModel."""
        respx.get(f"{IMPORTS_URL}/101").mock(
            return_value=httpx.Response(
                200, json=_make_import_response(id="101", status="InProgress")
            ),
        )
        idx = _make_async_index()
        result = await idx.describe_import("101")

        assert isinstance(result, ImportModel)
        assert result.id == "101"
        assert result.status == "InProgress"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_describe_import_int_id(self) -> None:
        """Verify int-to-str coercion."""
        route = respx.get(f"{IMPORTS_URL}/101").mock(
            return_value=httpx.Response(200, json=_make_import_response(id="101")),
        )
        idx = _make_async_index()
        result = await idx.describe_import(101)

        assert isinstance(result, ImportModel)
        assert result.id == "101"
        request = route.calls.last.request
        assert str(request.url).endswith("/bulk/imports/101")

    @pytest.mark.asyncio
    async def test_async_describe_import_invalid_id(self) -> None:
        """Verify empty string raises ValidationError."""
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="import id"):
            await idx.describe_import("")


# ---------------------------------------------------------------------------
# cancel_import
# ---------------------------------------------------------------------------


class TestAsyncCancelImport:
    """Tests for AsyncIndex.cancel_import()."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_cancel_import(self) -> None:
        """Mock DELETE, verify returns None."""
        respx.delete(f"{IMPORTS_URL}/101").mock(
            return_value=httpx.Response(202),
        )
        idx = _make_async_index()
        result = await idx.cancel_import("101")

        assert result is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_cancel_import_int_id(self) -> None:
        """Verify cancel_import(42) converts to '42'."""
        route = respx.delete(f"{IMPORTS_URL}/42").mock(
            return_value=httpx.Response(202),
        )
        idx = _make_async_index()
        result = await idx.cancel_import(42)

        assert result is None
        request = route.calls.last.request
        assert str(request.url).endswith("/bulk/imports/42")


# ---------------------------------------------------------------------------
# list_imports (auto-paginating async generator)
# ---------------------------------------------------------------------------


class TestAsyncListImports:
    """Tests for AsyncIndex.list_imports()."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_list_imports(self) -> None:
        """Mock paginated responses, verify async iteration yields all items."""
        page1_imports = [
            _make_import_response(id="imp-1"),
            _make_import_response(id="imp-2"),
        ]
        page2_imports = [
            _make_import_response(id="imp-3"),
        ]
        respx.get(IMPORTS_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_make_list_response(page1_imports, pagination_token="tok-abc"),
                ),
                httpx.Response(
                    200,
                    json=_make_list_response(page2_imports),
                ),
            ],
        )
        idx = _make_async_index()
        results: list[ImportModel] = [item async for item in idx.list_imports()]

        assert len(results) == 3
        assert [r.id for r in results] == ["imp-1", "imp-2", "imp-3"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_list_imports_single_page(self) -> None:
        """Mock single page response, verify all items yielded."""
        imports = [
            _make_import_response(id="imp-1", status="Completed"),
            _make_import_response(id="imp-2", status="Pending"),
        ]
        respx.get(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response(imports)),
        )
        idx = _make_async_index()
        results: list[ImportModel] = [item async for item in idx.list_imports()]

        assert len(results) == 2
        assert all(isinstance(r, ImportModel) for r in results)
        assert results[0].id == "imp-1"
        assert results[1].id == "imp-2"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_list_imports_empty(self) -> None:
        """Mock empty response, verify yields nothing."""
        respx.get(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response([])),
        )
        idx = _make_async_index()
        results: list[ImportModel] = [item async for item in idx.list_imports()]

        assert results == []


# ---------------------------------------------------------------------------
# list_imports_paginated (single page)
# ---------------------------------------------------------------------------


class TestAsyncListImportsPaginated:
    """Tests for AsyncIndex.list_imports_paginated()."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_list_imports_paginated(self) -> None:
        """Verify returns ImportList."""
        imports = [
            _make_import_response(id="imp-1"),
            _make_import_response(id="imp-2"),
        ]
        respx.get(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response(imports)),
        )
        idx = _make_async_index()
        result = await idx.list_imports_paginated()

        assert isinstance(result, ImportList)
        assert len(result) == 2
        assert result[0].id == "imp-1"
        assert result[1].id == "imp-2"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_list_imports_paginated_with_params(self) -> None:
        """Verify limit and paginationToken appear in query params."""
        route = respx.get(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response([])),
        )
        idx = _make_async_index()
        await idx.list_imports_paginated(limit=5, pagination_token="tok-xyz")

        request = route.calls.last.request
        assert request.url.params["limit"] == "5"
        assert request.url.params["paginationToken"] == "tok-xyz"
