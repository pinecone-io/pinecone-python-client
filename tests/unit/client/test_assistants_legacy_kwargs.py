"""Unit tests for legacy kwarg aliases on Assistants.create."""

from __future__ import annotations

import pytest

from pinecone.client.assistants import Assistants
from pinecone.errors.exceptions import PineconeValueError


def test_create_accepts_legacy_assistant_name_kwarg(mock_assistants: Assistants) -> None:
    """assistant_name= is accepted as a legacy alias for name=."""
    result = mock_assistants.create(assistant_name="legacy-name")
    assert result.name == "legacy-name"


def test_create_rejects_both_name_and_assistant_name(mock_assistants: Assistants) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        mock_assistants.create(name="a", assistant_name="b")


def test_create_rejects_unknown_kwarg(mock_assistants: Assistants) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        mock_assistants.create(name="foo", bogus_param=1)


def test_create_missing_name_raises(mock_assistants: Assistants) -> None:
    """Calling create() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        mock_assistants.create()


def test_create_with_name_still_works(mock_assistants: Assistants) -> None:
    """The canonical name= parameter continues to work as before."""
    result = mock_assistants.create(name="my-assistant")
    assert result.name == "legacy-name"  # canned response from fixture
