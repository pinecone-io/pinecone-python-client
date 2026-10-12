"""Annotations on the schema builder and documents surface resolve at runtime."""

from __future__ import annotations

import inspect
import subprocess
import sys
import typing

import pytest

from pinecone.async_client.documents import AsyncDocuments
from pinecone.client.documents import Documents
from pinecone.schema_builder import SchemaBuilder

_CLASSES = [SchemaBuilder, Documents, AsyncDocuments]


def _public_callables() -> list[tuple[str, typing.Any]]:
    found = []
    for cls in _CLASSES:
        for name, member in inspect.getmembers(cls, callable):
            if name.startswith("_") or not inspect.isfunction(member):
                continue
            found.append((f"{cls.__name__}.{name}", member))
    return found


@pytest.mark.parametrize(
    ("label", "func"), _public_callables(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_get_type_hints_resolves(label: str, func: typing.Any) -> None:
    typing.get_type_hints(func)


def test_add_string_field_embed_hints_name_the_config_classes() -> None:
    from pinecone.models.indexes.schema import FieldEmbedConfig, FieldSparseEmbedConfig

    hints = typing.get_type_hints(SchemaBuilder.add_string_field)
    assert FieldEmbedConfig in typing.get_args(hints["embed"])
    assert FieldSparseEmbedConfig in typing.get_args(hints["sparse_embed"])


def test_schema_builder_imports_standalone_without_a_cycle() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import pinecone.schema_builder"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
