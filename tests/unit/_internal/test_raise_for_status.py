"""Tests for _raise_for_status error message extraction."""
from __future__ import annotations

import httpx
import pytest

from pinecone._internal.http_client import _raise_for_status
from pinecone.errors.exceptions import ApiError


def _resp(status: int, body: bytes, ct: str = "application/json") -> httpx.Response:
    return httpx.Response(status_code=status, content=body, headers={"content-type": ct})


def test_extracts_message_key() -> None:
    with pytest.raises(ApiError) as exc_info:
        _raise_for_status(_resp(400, b'{"message": "Invalid field name"}'))
    assert "Invalid field name" in str(exc_info.value)


def test_extracts_error_key_as_fallback() -> None:
    with pytest.raises(ApiError) as exc_info:
        _raise_for_status(_resp(400, b'{"error": "Unsupported operation"}'))
    assert "Unsupported operation" in str(exc_info.value)


def test_extracts_detail_key_as_fallback() -> None:
    with pytest.raises(ApiError) as exc_info:
        _raise_for_status(_resp(422, b'{"detail": "Field validation failed"}'))
    assert "Field validation failed" in str(exc_info.value)


def test_includes_body_content_when_no_known_key() -> None:
    with pytest.raises(ApiError) as exc_info:
        _raise_for_status(_resp(400, b'{"code": 400, "status": "INVALID_ARGUMENT"}'))
    assert "INVALID_ARGUMENT" in str(exc_info.value)


def test_includes_plain_text_body_when_not_json() -> None:
    with pytest.raises(ApiError) as exc_info:
        _raise_for_status(_resp(400, b"Invalid query syntax near '('", ct="text/plain"))
    assert "Invalid query syntax" in str(exc_info.value)


def test_success_does_not_raise() -> None:
    _raise_for_status(_resp(200, b'{"result": "ok"}'))
