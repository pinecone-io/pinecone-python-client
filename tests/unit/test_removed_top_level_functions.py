from __future__ import annotations

import pytest

import pinecone

REMOVED_NAMES = [
    "init",
    "create_index",
    "delete_index",
    "list_indexes",
    "describe_index",
    "configure_index",
    "scale_index",
    "create_collection",
    "delete_collection",
    "describe_collection",
    "list_collections",
]


@pytest.mark.parametrize("name", REMOVED_NAMES)
def test_removed_function_raises_attribute_error(name: str) -> None:
    with pytest.raises(AttributeError) as exc_info:
        getattr(pinecone, name)
    msg = str(exc_info.value)
    assert name in msg
    assert "no longer a top-level attribute of the pinecone package" in msg
    assert "Example:" in msg


def test_removed_function_calls_raise_attribute_error() -> None:
    with pytest.raises(AttributeError):
        pinecone.init()  # type: ignore[attr-defined]
