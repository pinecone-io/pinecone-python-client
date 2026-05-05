"""Unit tests for Pinecone.create_index backcompat shim — schema parameter forwarding."""

from __future__ import annotations

from unittest.mock import MagicMock

from pinecone import Pinecone, ServerlessSpec


def _make_pc_with_mock_indexes() -> tuple[Pinecone, MagicMock]:
    pc = Pinecone(api_key="test-key")
    mock_indexes = MagicMock()
    mock_indexes.create = MagicMock(return_value=MagicMock())
    pc._indexes = mock_indexes
    return pc, mock_indexes


def test_create_index_shim_forwards_schema() -> None:
    """Shim must forward schema kwarg to Indexes.create."""
    pc, mock_indexes = _make_pc_with_mock_indexes()
    pc.create_index(
        name="test",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        dimension=1536,
        schema={"field": {"type": "str"}},
    )

    mock_indexes.create.assert_called_once()
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["schema"] == {"field": {"type": "str"}}


def test_create_index_shim_schema_defaults_to_none() -> None:
    """schema defaults to None when not passed, and None is forwarded to Indexes.create."""
    pc, mock_indexes = _make_pc_with_mock_indexes()
    pc.create_index(
        name="test",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        dimension=1536,
    )

    mock_indexes.create.assert_called_once()
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["schema"] is None
