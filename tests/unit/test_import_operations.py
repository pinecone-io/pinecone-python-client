"""Unit tests for Index.start_import, describe_import, cancel_import methods."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from pinecone import Index
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


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# start_import
# ---------------------------------------------------------------------------


class TestStartImport:
    """Tests for Index.start_import()."""

    @respx.mock
    def test_start_import_default_error_mode(self) -> None:
        """Verify body has errorMode.onError = 'continue' by default."""
        route = respx.post(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-001"}),
        )
        idx = _make_index()
        result = idx.start_import(uri="s3://bucket/data/")

        assert isinstance(result, StartImportResponse)
        assert result.id == "import-001"

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["uri"] == "s3://bucket/data/"
        assert body["errorMode"] == {"onError": "continue"}

    @respx.mock
    def test_start_import_abort_mode(self) -> None:
        """Verify error_mode='ABORT' normalizes to 'abort' in body."""
        route = respx.post(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-002"}),
        )
        idx = _make_index()
        idx.start_import(uri="s3://bucket/data/", error_mode="ABORT")

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["errorMode"] == {"onError": "abort"}

    def test_start_import_invalid_error_mode(self) -> None:
        """Verify error_mode='skip' raises ValidationError."""
        idx = _make_index()
        with pytest.raises(ValidationError, match="error_mode"):
            idx.start_import(uri="s3://bucket/data/", error_mode="skip")

    @respx.mock
    def test_start_import_with_integration_id(self) -> None:
        """Verify integrationId appears in body when set."""
        route = respx.post(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-003"}),
        )
        idx = _make_index()
        idx.start_import(uri="s3://bucket/data/", integration_id="int-456")

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["integrationId"] == "int-456"

    @respx.mock
    def test_start_import_no_integration_id(self) -> None:
        """Verify integrationId absent from body when not set."""
        route = respx.post(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "import-004"}),
        )
        idx = _make_index()
        idx.start_import(uri="s3://bucket/data/")

        request = route.calls.last.request
        body = json.loads(request.content)
        assert "integrationId" not in body


# ---------------------------------------------------------------------------
# describe_import
# ---------------------------------------------------------------------------


class TestDescribeImport:
    """Tests for Index.describe_import()."""

    @respx.mock
    def test_describe_import(self) -> None:
        """Mock GET /bulk/imports/101, verify returns ImportModel."""
        respx.get(f"{IMPORTS_URL}/101").mock(
            return_value=httpx.Response(
                200, json=_make_import_response(id="101", status="InProgress")
            ),
        )
        idx = _make_index()
        result = idx.describe_import("101")

        assert isinstance(result, ImportModel)
        assert result.id == "101"
        assert result.status == "InProgress"

    @respx.mock
    def test_describe_import_int_id(self) -> None:
        """Verify describe_import(101) converts to '101'."""
        route = respx.get(f"{IMPORTS_URL}/101").mock(
            return_value=httpx.Response(
                200, json=_make_import_response(id="101")
            ),
        )
        idx = _make_index()
        result = idx.describe_import(101)

        assert isinstance(result, ImportModel)
        assert result.id == "101"
        # Verify the URL path used the string version
        request = route.calls.last.request
        assert str(request.url).endswith("/bulk/imports/101")

    def test_describe_import_invalid_id(self) -> None:
        """Verify empty string raises ValidationError."""
        idx = _make_index()
        with pytest.raises(ValidationError, match="import id"):
            idx.describe_import("")


# ---------------------------------------------------------------------------
# cancel_import
# ---------------------------------------------------------------------------


class TestCancelImport:
    """Tests for Index.cancel_import()."""

    @respx.mock
    def test_cancel_import(self) -> None:
        """Mock DELETE /bulk/imports/101, verify returns None."""
        respx.delete(f"{IMPORTS_URL}/101").mock(
            return_value=httpx.Response(202),
        )
        idx = _make_index()
        result = idx.cancel_import("101")

        assert result is None

    @respx.mock
    def test_cancel_import_int_id(self) -> None:
        """Verify cancel_import(42) converts to '42'."""
        route = respx.delete(f"{IMPORTS_URL}/42").mock(
            return_value=httpx.Response(202),
        )
        idx = _make_index()
        result = idx.cancel_import(42)

        assert result is None
        request = route.calls.last.request
        assert str(request.url).endswith("/bulk/imports/42")


# ---------------------------------------------------------------------------
# Shared validation: _validate_import_id
# ---------------------------------------------------------------------------


class TestValidateImportId:
    """Edge cases for import ID validation."""

    def test_id_too_long_raises(self) -> None:
        """IDs over 1000 characters are rejected."""
        idx = _make_index()
        with pytest.raises(ValidationError, match="import id"):
            idx.describe_import("x" * 1001)

    def test_id_at_max_length_ok(self) -> None:
        """IDs exactly 1000 characters are accepted (validation only, no HTTP)."""
        idx = _make_index()
        # This will fail at HTTP level since we're not mocking, but validation passes
        long_id = "x" * 1000
        result = idx._validate_import_id(long_id)
        assert result == long_id

    def test_int_zero_converts_to_string(self) -> None:
        """Integer 0 converts to '0' which is valid (length 1)."""
        idx = _make_index()
        result = idx._validate_import_id(0)
        assert result == "0"


# ---------------------------------------------------------------------------
# list_imports (auto-paginating generator)
# ---------------------------------------------------------------------------


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


class TestListImports:
    """Tests for Index.list_imports()."""

    @respx.mock
    def test_list_imports_single_page(self) -> None:
        """Mock GET returning 2 imports with no pagination token, verify yields 2 ImportModel items."""
        imports = [
            _make_import_response(id="imp-1", status="Completed"),
            _make_import_response(id="imp-2", status="Pending"),
        ]
        respx.get(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response(imports)),
        )
        idx = _make_index()
        results = list(idx.list_imports())

        assert len(results) == 2
        assert all(isinstance(r, ImportModel) for r in results)
        assert results[0].id == "imp-1"
        assert results[1].id == "imp-2"

    @respx.mock
    def test_list_imports_multi_page(self) -> None:
        """Mock GET returning page 1 with pagination token, then page 2 without; verify all items yielded in order."""
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
        idx = _make_index()
        results = list(idx.list_imports())

        assert len(results) == 3
        assert [r.id for r in results] == ["imp-1", "imp-2", "imp-3"]

    @respx.mock
    def test_list_imports_with_limit(self) -> None:
        """Verify limit=10 appears in query params."""
        route = respx.get(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response([])),
        )
        idx = _make_index()
        list(idx.list_imports(limit=10))

        request = route.calls.last.request
        assert request.url.params["limit"] == "10"

    @respx.mock
    def test_list_imports_empty(self) -> None:
        """Mock empty response, verify yields nothing."""
        respx.get(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response([])),
        )
        idx = _make_index()
        results = list(idx.list_imports())

        assert results == []


# ---------------------------------------------------------------------------
# list_imports_paginated (single page)
# ---------------------------------------------------------------------------


class TestListImportsPaginated:
    """Tests for Index.list_imports_paginated()."""

    @respx.mock
    def test_list_imports_paginated(self) -> None:
        """Verify returns ImportList (not a generator)."""
        imports = [
            _make_import_response(id="imp-1"),
            _make_import_response(id="imp-2"),
        ]
        respx.get(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response(imports)),
        )
        idx = _make_index()
        result = idx.list_imports_paginated()

        assert isinstance(result, ImportList)
        assert len(result) == 2
        assert result[0].id == "imp-1"
        assert result[1].id == "imp-2"

    @respx.mock
    def test_list_imports_paginated_with_params(self) -> None:
        """Verify limit and paginationToken appear in query params."""
        route = respx.get(IMPORTS_URL).mock(
            return_value=httpx.Response(200, json=_make_list_response([])),
        )
        idx = _make_index()
        idx.list_imports_paginated(limit=5, pagination_token="tok-xyz")

        request = route.calls.last.request
        assert request.url.params["limit"] == "5"
        assert request.url.params["paginationToken"] == "tok-xyz"
