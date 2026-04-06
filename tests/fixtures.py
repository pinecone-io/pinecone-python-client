"""respx mock helpers for common Pinecone API interactions.

Each helper accepts a ``respx.MockRouter`` and wires up routes that return
realistic responses from ``tests.factories``.
"""

from __future__ import annotations

import httpx
import respx

from tests.factories import (
    make_backup_response,
    make_collection_response,
    make_describe_index_stats_response,
    make_embed_response,
    make_fetch_response,
    make_index_list_response,
    make_index_response,
    make_query_response,
    make_rerank_response,
    make_upsert_response,
)

CONTROL_BASE = "https://api.test.pinecone.io"
DATA_BASE = "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INFERENCE_BASE = "https://api.test.pinecone.io"


def mock_control_plane(router: respx.MockRouter) -> respx.MockRouter:
    """Set up respx routes for index CRUD on the control plane.

    Routes:
        GET  /indexes           -> IndexList
        POST /indexes           -> IndexModel
        GET  /indexes/test-index -> IndexModel
        DELETE /indexes/test-index -> 204
        GET  /collections/test-collection -> CollectionModel
        GET  /indexes/test-index/backups/test-backup -> BackupModel
    """
    router.get(f"{CONTROL_BASE}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response()),
    )
    router.post(f"{CONTROL_BASE}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )
    router.get(f"{CONTROL_BASE}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )
    router.delete(f"{CONTROL_BASE}/indexes/test-index").mock(
        return_value=httpx.Response(204),
    )
    router.get(f"{CONTROL_BASE}/collections/test-collection").mock(
        return_value=httpx.Response(200, json=make_collection_response()),
    )
    router.get(f"{CONTROL_BASE}/indexes/test-index/backups/test-backup").mock(
        return_value=httpx.Response(200, json=make_backup_response()),
    )
    return router


def mock_data_plane(router: respx.MockRouter) -> respx.MockRouter:
    """Set up respx routes for vector operations on the data plane.

    Routes:
        POST /vectors/upsert  -> UpsertResponse
        POST /query           -> QueryResponse
        GET  /vectors/fetch   -> FetchResponse
    """
    router.post(f"{DATA_BASE}/vectors/upsert").mock(
        return_value=httpx.Response(200, json=make_upsert_response()),
    )
    router.post(f"{DATA_BASE}/query").mock(
        return_value=httpx.Response(200, json=make_query_response()),
    )
    router.get(f"{DATA_BASE}/vectors/fetch").mock(
        return_value=httpx.Response(200, json=make_fetch_response()),
    )
    return router


def mock_inference(router: respx.MockRouter) -> respx.MockRouter:
    """Set up respx routes for embed/rerank on the inference API.

    Routes:
        POST /embed  -> EmbeddingsList
        POST /rerank -> RerankResult
    """
    router.post(f"{INFERENCE_BASE}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )
    router.post(f"{INFERENCE_BASE}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )
    return router
