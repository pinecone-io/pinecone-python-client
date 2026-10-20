"""A malformed PINECONE_ADDITIONAL_HEADERS warns once and never logs the value."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Iterator

import pytest

from pinecone import Pinecone
from pinecone._internal.config import _parse_additional_headers
from pinecone.config.pinecone_config import PineconeConfig as LegacyPineconeConfig

_SECRET = "SENTINEL-SECRET-8c1f"
_MALFORMED = '{"Authorization": "Bearer ' + _SECRET + '"'


@pytest.fixture(autouse=True)
def _fresh_parse_cache() -> Iterator[None]:
    _parse_additional_headers.cache_clear()
    yield
    _parse_additional_headers.cache_clear()


def test_malformed_env_warns_once_without_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", _MALFORMED)
    with caplog.at_level(logging.WARNING):
        pc = Pinecone(api_key="k")
        pc.index(host="https://idx.example")
        Pinecone(api_key="k")
        LegacyPineconeConfig.build(api_key="k")

    assert pc.config.additional_headers == {}
    warnings = [r for r in caplog.records if "PINECONE_ADDITIONAL_HEADERS" in r.getMessage()]
    assert len(warnings) == 1
    assert "JSONDecodeError" in warnings[0].getMessage()
    assert _SECRET not in caplog.text


def test_malformed_env_on_stderr_without_logging_configured() -> None:
    env = {**os.environ, "PINECONE_API_KEY": "x", "PINECONE_ADDITIONAL_HEADERS": _MALFORMED}
    result = subprocess.run(
        [sys.executable, "-c", "from pinecone import Pinecone; Pinecone()"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    assert _SECRET not in result.stderr
    assert result.stderr.count("Failed to parse PINECONE_ADDITIONAL_HEADERS") == 1
