"""Response factories that return realistic dicts matching the Pinecone API specs.

Each factory returns a dict suitable for ``httpx.Response(200, json=factory())``.
Pass ``**overrides`` to customise individual fields.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Control-plane factories
# ---------------------------------------------------------------------------


def make_index_response(**overrides: Any) -> dict[str, Any]:
    """Return a single IndexModel dict (db_control ``GET /indexes/{name}``)."""
    base: dict[str, Any] = {
        "name": "test-index",
        "dimension": 1536,
        "metric": "cosine",
        "host": "test-index-abc1234.svc.us-east1-gcp.pinecone.io",
        "deletion_protection": "disabled",
        "tags": {},
        "spec": {
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1",
                "read_capacity": {
                    "mode": "OnDemand",
                    "status": {"state": "Ready"},
                },
            },
        },
        "status": {"ready": True, "state": "Ready"},
        "vector_type": "dense",
    }
    base.update(overrides)
    return base


def make_index_list_response(**overrides: Any) -> dict[str, Any]:
    """Return an IndexList dict (db_control ``GET /indexes``)."""
    base: dict[str, Any] = {
        "indexes": [make_index_response()],
    }
    base.update(overrides)
    return base


def make_collection_response(**overrides: Any) -> dict[str, Any]:
    """Return a single CollectionModel dict (db_control ``GET /collections/{name}``)."""
    base: dict[str, Any] = {
        "name": "test-collection",
        "size": 10_000_000,
        "status": "Ready",
        "dimension": 1536,
        "vector_count": 120_000,
        "environment": "us-east1-gcp",
    }
    base.update(overrides)
    return base


def make_backup_response(**overrides: Any) -> dict[str, Any]:
    """Return a single BackupModel dict (db_control ``GET /indexes/{name}/backups/{id}``)."""
    base: dict[str, Any] = {
        "backup_id": "670e8400-e29b-41d4-a716-446655440001",
        "source_index_name": "test-index",
        "source_index_id": "670e8400-e29b-41d4-a716-446655440000",
        "name": "backup-2025-02-04",
        "description": "Backup before bulk update.",
        "status": "Ready",
        "cloud": "aws",
        "region": "us-east-1",
        "dimension": 1536,
        "metric": "cosine",
        "record_count": 120_000,
        "namespace_count": 3,
        "size_bytes": 10_000_000,
        "tags": {},
        "created_at": "2025-02-04T12:00:00Z",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Data-plane factories
# ---------------------------------------------------------------------------


def make_upsert_response(**overrides: Any) -> dict[str, Any]:
    """Return an UpsertResponse dict (db_data ``POST /vectors/upsert``)."""
    base: dict[str, Any] = {
        "upsertedCount": 10,
    }
    base.update(overrides)
    return base


def make_query_response(**overrides: Any) -> dict[str, Any]:
    """Return a QueryResponse dict (db_data ``POST /query``)."""
    base: dict[str, Any] = {
        "matches": [
            {
                "id": "vec-1",
                "score": 0.95,
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "documentary"},
            },
            {
                "id": "vec-2",
                "score": 0.80,
                "values": [0.4, 0.5, 0.6],
                "metadata": {"genre": "comedy"},
            },
        ],
        "namespace": "test-namespace",
        "usage": {"readUnits": 5},
    }
    base.update(overrides)
    return base


def make_fetch_response(**overrides: Any) -> dict[str, Any]:
    """Return a FetchResponse dict (db_data ``GET /vectors/fetch``)."""
    base: dict[str, Any] = {
        "vectors": {
            "id-1": {"id": "id-1", "values": [1.0, 1.5]},
            "id-2": {"id": "id-2", "values": [2.0, 1.0]},
        },
        "namespace": "test-namespace",
        "usage": {"readUnits": 1},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Inference factories
# ---------------------------------------------------------------------------


def make_embed_response(**overrides: Any) -> dict[str, Any]:
    """Return an EmbeddingsList dict (inference ``POST /embed``)."""
    base: dict[str, Any] = {
        "model": "multilingual-e5-large",
        "vector_type": "dense",
        "data": [
            {"values": [0.1, 0.2, 0.3], "vector_type": "dense"},
        ],
        "usage": {"total_tokens": 205},
    }
    base.update(overrides)
    return base


def make_rerank_response(**overrides: Any) -> dict[str, Any]:
    """Return a RerankResult dict (inference ``POST /rerank``)."""
    base: dict[str, Any] = {
        "model": "bge-reranker-v2-m3",
        "data": [
            {
                "index": 0,
                "score": 0.95,
                "document": {"id": "1", "text": "Paris is the capital of France."},
            },
            {
                "index": 1,
                "score": 0.45,
                "document": {"id": "2", "text": "Berlin is the capital of Germany."},
            },
        ],
        "usage": {"rerank_units": 1},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Error factory
# ---------------------------------------------------------------------------


def make_error_response(
    status_code: int = 500,
    message: str = "Internal server error",
    **overrides: Any,
) -> dict[str, Any]:
    """Return an ErrorResponse dict matching the Pinecone error envelope."""
    code_map: dict[int, str] = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "ALREADY_EXISTS",
        429: "QUOTA_EXCEEDED",
        500: "UNKNOWN",
    }
    base: dict[str, Any] = {
        "status": status_code,
        "error": {
            "code": code_map.get(status_code, "UNKNOWN"),
            "message": message,
        },
    }
    base.update(overrides)
    return base
