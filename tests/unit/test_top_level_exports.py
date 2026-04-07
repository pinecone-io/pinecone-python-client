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


def test_new_response_types_importable_from_pinecone() -> None:
    from pinecone import (
        BackupList,
        BackupModel,
        CreateIndexFromBackupResponse,
        EmbeddingsList,
        FetchByMetadataResponse,
        ImportList,
        ImportModel,
        ListNamespacesResponse,
        ModelInfo,
        ModelInfoList,
        NamespaceDescription,
        QueryNamespacesResults,
        RerankResult,
        RestoreJobList,
        RestoreJobModel,
        SearchRecordsResponse,
        StartImportResponse,
        UpsertRecordsResponse,
    )

    assert BackupList.__name__ == "BackupList"
    assert BackupModel.__name__ == "BackupModel"
    assert CreateIndexFromBackupResponse.__name__ == "CreateIndexFromBackupResponse"
    assert EmbeddingsList.__name__ == "EmbeddingsList"
    assert FetchByMetadataResponse.__name__ == "FetchByMetadataResponse"
    assert ImportList.__name__ == "ImportList"
    assert ImportModel.__name__ == "ImportModel"
    assert ListNamespacesResponse.__name__ == "ListNamespacesResponse"
    assert ModelInfo.__name__ == "ModelInfo"
    assert ModelInfoList.__name__ == "ModelInfoList"
    assert NamespaceDescription.__name__ == "NamespaceDescription"
    assert QueryNamespacesResults.__name__ == "QueryNamespacesResults"
    assert RerankResult.__name__ == "RerankResult"
    assert RestoreJobList.__name__ == "RestoreJobList"
    assert RestoreJobModel.__name__ == "RestoreJobModel"
    assert SearchRecordsResponse.__name__ == "SearchRecordsResponse"
    assert StartImportResponse.__name__ == "StartImportResponse"
    assert UpsertRecordsResponse.__name__ == "UpsertRecordsResponse"


def test_model_types_in_all() -> None:
    import pinecone

    expected = {
        "BackupList",
        "BackupModel",
        "CollectionList",
        "CollectionModel",
        "CreateIndexFromBackupResponse",
        "DescribeIndexStatsResponse",
        "EmbeddingsList",
        "FetchByMetadataResponse",
        "FetchResponse",
        "ImportList",
        "ImportModel",
        "IndexList",
        "IndexModel",
        "ListNamespacesResponse",
        "ListResponse",
        "ModelInfo",
        "ModelInfoList",
        "NamespaceDescription",
        "PodSpec",
        "QueryNamespacesResults",
        "QueryResponse",
        "RerankResult",
        "RestoreJobList",
        "RestoreJobModel",
        "SearchRecordsResponse",
        "ServerlessSpec",
        "SparseValues",
        "StartImportResponse",
        "UpdateResponse",
        "UpsertRecordsResponse",
        "UpsertResponse",
        "Vector",
    }
    assert expected.issubset(set(pinecone.__all__))
