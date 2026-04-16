"""Unit tests for Preview and AsyncPreview namespace classes."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pinecone.preview.schema_builder
from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.preview import AsyncPreview, Preview, SchemaBuilder


def _cfg() -> PineconeConfig:
    return PineconeConfig(api_key="test")


def _http() -> HTTPClient:
    return cast(HTTPClient, MagicMock())


def _async_http() -> AsyncHTTPClient:
    return cast(AsyncHTTPClient, MagicMock())


def test_preview_constructor_stores_config() -> None:
    cfg = _cfg()
    preview = Preview(http=_http(), config=cfg)
    assert preview._config is cfg


def test_async_preview_constructor_stores_config() -> None:
    cfg = _cfg()
    preview = AsyncPreview(http=_async_http(), config=cfg)
    assert preview._config is cfg


def test_preview_repr() -> None:
    assert repr(Preview(http=_http(), config=_cfg())) == "Preview()"


def test_async_preview_repr() -> None:
    assert repr(AsyncPreview(http=_async_http(), config=_cfg())) == "AsyncPreview()"


def test_preview_and_async_preview_are_distinct_types() -> None:
    assert Preview is not AsyncPreview  # type: ignore[comparison-overlap]
    assert not issubclass(Preview, AsyncPreview)
    assert not issubclass(AsyncPreview, Preview)


def test_schema_builder_reexported_from_preview_module() -> None:
    assert SchemaBuilder is pinecone.preview.schema_builder.SchemaBuilder
