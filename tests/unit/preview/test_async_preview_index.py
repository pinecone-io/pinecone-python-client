"""Unit tests covering error paths and __repr__ on AsyncPreviewIndex."""

from __future__ import annotations

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone.preview.async_index import AsyncPreviewIndex


async def _dummy_provider() -> str:
    return "resolved.svc.pinecone.io"


@pytest.fixture
def config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key", host="https://api.test.pinecone.io")


def test_host_raises_when_unresolved(config: PineconeConfig) -> None:
    # Created with a provider but host not yet resolved (_resolved_host is None).
    idx = AsyncPreviewIndex(config=config, _host_provider=_dummy_provider)
    with pytest.raises(RuntimeError, match="Host not yet resolved"):
        _ = idx.host


@pytest.mark.asyncio
async def test_resolve_host_raises_when_no_provider(config: PineconeConfig) -> None:
    idx = AsyncPreviewIndex(config=config, _host_provider=_dummy_provider)
    # Simulate a state where both resolved host and provider are absent.
    idx._host_provider = None
    with pytest.raises(RuntimeError, match="no host or host_provider configured"):
        await idx._resolve_host()


def test_repr_includes_resolved_host(config: PineconeConfig) -> None:
    idx = AsyncPreviewIndex(config=config, host="svc.test.pinecone.io")
    assert repr(idx) == "AsyncPreviewIndex(host='svc.test.pinecone.io')"


def test_repr_shows_none_when_unresolved(config: PineconeConfig) -> None:
    # Provider present but resolve not yet called — _resolved_host stays None.
    idx = AsyncPreviewIndex(config=config, _host_provider=_dummy_provider)
    assert repr(idx) == "AsyncPreviewIndex(host=None)"
