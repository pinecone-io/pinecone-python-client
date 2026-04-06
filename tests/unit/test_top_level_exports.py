"""Tests that key model types are importable from the top-level pinecone package."""

from __future__ import annotations


def test_model_types_importable_from_pinecone() -> None:
    from pinecone import (
        CollectionList,
        CollectionModel,
        IndexList,
        IndexModel,
        PodSpec,
        QueryResponse,
        ServerlessSpec,
    )

    # Verify the imports resolve to the correct classes
    assert CollectionList.__name__ == "CollectionList"
    assert CollectionModel.__name__ == "CollectionModel"
    assert IndexList.__name__ == "IndexList"
    assert IndexModel.__name__ == "IndexModel"
    assert PodSpec.__name__ == "PodSpec"
    assert QueryResponse.__name__ == "QueryResponse"
    assert ServerlessSpec.__name__ == "ServerlessSpec"


def test_model_types_in_all() -> None:
    import pinecone

    expected = {
        "CollectionList",
        "CollectionModel",
        "IndexList",
        "IndexModel",
        "PodSpec",
        "QueryResponse",
        "ServerlessSpec",
    }
    assert expected.issubset(set(pinecone.__all__))
