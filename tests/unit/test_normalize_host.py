"""Unit tests for normalize_host()."""
from __future__ import annotations

import pytest

from pinecone._internal.config import normalize_host


@pytest.mark.parametrize(
    ("input_host", "expected"),
    [
        # None / empty → empty string
        (None, ""),
        ("", ""),
        # Bare host → https:// prepended
        ("foo.io", "https://foo.io"),
        ("my-index-abc123.svc.pinecone.io", "https://my-index-abc123.svc.pinecone.io"),
        # Single scheme → preserved as-is
        ("https://foo.io", "https://foo.io"),
        ("http://localhost:8080", "http://localhost:8080"),
        ("https://prod-1-data.ke.pinecone.io", "https://prod-1-data.ke.pinecone.io"),
        # Double scheme — outer https, inner https
        ("https://https://foo.io", "https://foo.io"),
        # Double scheme — outer https, inner http (preserve inner http, e.g. localhost)
        ("https://http://localhost:8080", "http://localhost:8080"),
        # Double scheme — outer http, inner https
        ("http://https://foo.io", "https://foo.io"),
        # Double scheme — outer http, inner http
        ("http://http://localhost:8080", "http://localhost:8080"),
    ],
)
def test_normalize_host(input_host: str | None, expected: str) -> None:
    assert normalize_host(input_host) == expected
