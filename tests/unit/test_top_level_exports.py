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


def test_vector_and_response_types_importable_from_pinecone() -> None:
    from pinecone import (
        DescribeIndexStatsResponse,
        FetchResponse,
        ListResponse,
        SparseValues,
        UpdateResponse,
        UpsertResponse,
        Vector,
    )

    assert DescribeIndexStatsResponse.__name__ == "DescribeIndexStatsResponse"
    assert FetchResponse.__name__ == "FetchResponse"
    assert ListResponse.__name__ == "ListResponse"
    assert SparseValues.__name__ == "SparseValues"
    assert UpdateResponse.__name__ == "UpdateResponse"
    assert UpsertResponse.__name__ == "UpsertResponse"
    assert Vector.__name__ == "Vector"


def test_model_types_in_all() -> None:
    import pinecone

    expected = {
        "CollectionList",
        "CollectionModel",
        "DescribeIndexStatsResponse",
        "FetchResponse",
        "IndexList",
        "IndexModel",
        "ListResponse",
        "PodSpec",
        "QueryResponse",
        "ServerlessSpec",
        "SparseValues",
        "UpdateResponse",
        "UpsertResponse",
        "Vector",
    }
    assert expected.issubset(set(pinecone.__all__))
