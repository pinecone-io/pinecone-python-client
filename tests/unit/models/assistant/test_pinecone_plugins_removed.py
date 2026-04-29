"""Pin the contract that pinecone_plugins.assistant.* is no longer importable.

Removed in v9 — see docs/migration/v9-migration.md §8.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "pinecone_plugins",
        "pinecone_plugins.assistant",
        "pinecone_plugins.assistant.models",
        "pinecone_plugins.assistant.models.chat",
        "pinecone_plugins.assistant.models.chat_completion",
        "pinecone_plugins.assistant.models.context_responses",
        "pinecone_plugins.assistant.models.evaluation_responses",
        "pinecone_plugins.assistant.models.file_model",
        "pinecone_plugins.assistant.models.list_assistants_response",
        "pinecone_plugins.assistant.models.list_files_response",
        "pinecone_plugins.assistant.models.shared",
        "pinecone_plugins.assistant.assistant.assistant",
    ],
)
def test_legacy_plugin_import_path_removed(name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_canonical_assistant_namespace_still_works() -> None:
    # Sanity check: removing pinecone_plugins must not break the canonical surface.
    from pinecone import Pinecone
    from pinecone.models.assistant import (
        AlignmentResponse,
        AssistantFileModel,
        AssistantModel,
        ChatResponse,
        ContextOptions,
        FileModel,
        Message,
        StreamChatResponseMessageStart,
    )

    pc = Pinecone(api_key="dummy-key-for-import-test")
    # pc.assistant is a property — accessing it must not raise.
    _ = pc.assistant
    _ = pc.assistants
    # Class-name aliases still resolve (added by BC-0001).
    assert FileModel is AssistantFileModel
