"""Unit tests for AssistantModelLegacyMethodsMixin scaffolding (BC-0015)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_assistant_model_has_legacy_mixin() -> None:
    from pinecone.models.assistant._legacy_methods import AssistantModelLegacyMethodsMixin
    from pinecone.models.assistant.model import AssistantModel

    assert issubclass(AssistantModel, AssistantModelLegacyMethodsMixin)


def test_assistant_model_resolve_raises_when_no_ref() -> None:
    from pinecone.models.assistant.model import AssistantModel

    m = AssistantModel(name="foo", status="Ready")
    with pytest.raises(RuntimeError, match="no client reference"):
        m._resolve_assistants()


def test_attach_ref_allows_resolution() -> None:
    """Writing into __dict__ sets _assistants so _resolve_assistants returns it."""
    from pinecone.models.assistant.model import AssistantModel

    model = AssistantModel(name="test-assistant", status="Ready")
    fake_ns: MagicMock = MagicMock()

    # Simulate what _attach_ref does: bypass msgspec __setattr__ via __dict__
    model.__dict__["_assistants"] = fake_ns

    assert model._resolve_assistants() is fake_ns


def test_model_has_dict_enabled() -> None:
    """AssistantModel uses dict=True so instances have a __dict__ for the back-ref."""
    from pinecone.models.assistant.model import AssistantModel

    model = AssistantModel(name="dict-test", status="Ready")
    # dict=True means the instance has a __dict__
    assert hasattr(model, "__dict__")


def test_struct_fields_unaffected_by_back_ref() -> None:
    """Setting _assistants does not interfere with struct field access."""
    from pinecone.models.assistant.model import AssistantModel

    model = AssistantModel(name="field-test", status="Initializing")
    fake_ns: MagicMock = MagicMock()
    model.__dict__["_assistants"] = fake_ns

    assert model.name == "field-test"
    assert model.status == "Initializing"
    assert model._resolve_assistants() is fake_ns
