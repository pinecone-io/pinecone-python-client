"""Unit tests for AssistantsLegacyNamespaceMixin.create_assistant."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone.client.assistants import Assistants


def test_create_assistant_legacy_method(mock_assistants: Assistants) -> None:
    """create_assistant() with assistant_name= delegates to create() and returns a model."""
    result = mock_assistants.create_assistant(assistant_name="foo", instructions="be nice")
    # Response is the canned fixture value from mock_assistants
    assert result.name == "legacy-name"


def test_create_assistant_legacy_forwards_to_create(
    mock_assistants: Assistants, spy_create: MagicMock
) -> None:
    """create_assistant() forwards the right arguments to create()."""
    mock_assistants.create_assistant(assistant_name="foo")
    spy_create.assert_called_once_with(
        name="foo",
        instructions=None,
        metadata=None,
        region="us",
        timeout=None,
    )


def test_create_assistant_with_name_kwarg(mock_assistants: Assistants) -> None:
    """create_assistant() accepts the current-style name= kwarg too."""
    result = mock_assistants.create_assistant(name="bar")
    assert result.name == "legacy-name"


def test_create_assistant_name_kwarg_forwards_correctly(
    mock_assistants: Assistants, spy_create: MagicMock
) -> None:
    """create_assistant(name=...) resolves to create(name=...)."""
    mock_assistants.create_assistant(name="bar")
    spy_create.assert_called_once_with(
        name="bar",
        instructions=None,
        metadata=None,
        region="us",
        timeout=None,
    )


def test_create_assistant_no_name_propagates_error(
    mock_assistants: Assistants,
) -> None:
    """create_assistant() with no name propagates the error from create()."""
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="missing required"):
        mock_assistants.create_assistant()
