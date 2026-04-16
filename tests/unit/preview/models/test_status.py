"""Unit tests for PreviewIndexStatus model."""

from __future__ import annotations

import msgspec

from pinecone.preview.models.status import PreviewIndexStatus


def test_index_status_fields() -> None:
    status = PreviewIndexStatus(ready=True, state="Ready")
    assert status.ready is True
    assert status.state == "Ready"


def test_index_status_decode() -> None:
    raw = b'{"ready": false, "state": "Initializing"}'
    status = msgspec.json.decode(raw, type=PreviewIndexStatus)
    assert status.ready is False
    assert status.state == "Initializing"
