"""Unit tests for the GrpcIndex wrapper class."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PineconeConnectionError,
    PineconeTimeoutError,
    PineconeValueError,
    ServiceError,
    UnauthorizedError,
    ValidationError,
)
from pinecone.grpc import GrpcIndex
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import SearchRecordsResponse
from pinecone.models.vectors.vector import Vector

# The GrpcChannel import is a lazy import inside __init__, so we need to
# create a fake module to patch into sys.modules.
_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index(mock_channel: MagicMock, **kwargs: Any) -> GrpcIndex:
    """Helper to create a GrpcIndex with a mocked GrpcChannel."""
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host=kwargs.pop("host", "test-index-abc123.svc.pinecone.io"),
            api_key=kwargs.pop("api_key", "test-api-key"),
            **kwargs,
        )


@pytest.fixture()
def mock_channel() -> MagicMock:
    """Create a mock GrpcChannel."""
    return MagicMock()


@pytest.fixture()
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    """Create a GrpcIndex with a mocked GrpcChannel."""
    return _make_grpc_index(mock_channel)


class TestGrpcIndexInit:
    """Tests for GrpcIndex initialization."""

    def test_init_with_explicit_api_key(self, mock_channel: MagicMock) -> None:
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            idx = GrpcIndex(
                host="test-index.svc.pinecone.io",
                api_key="my-key",
            )
        assert idx.host == "https://test-index.svc.pinecone.io"
        mock_module.GrpcChannel.assert_called_once()
        call_args = mock_module.GrpcChannel.call_args
        assert call_args[0][1] == "my-key"  # api_key

    def test_init_with_env_api_key(
        self, mock_channel: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-key")
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            GrpcIndex(host="test-index.svc.pinecone.io")
        call_args = mock_module.GrpcChannel.call_args
        assert call_args[0][1] == "env-key"

    def test_init_no_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="No API key provided"):
            GrpcIndex(host="test-index.svc.pinecone.io")

    def test_init_empty_host_raises(self) -> None:
        with pytest.raises(ValidationError):
            GrpcIndex(host="", api_key="test-key")

    def test_init_custom_timeouts(self, mock_channel: MagicMock) -> None:
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            GrpcIndex(
                host="test-index.svc.pinecone.io",
                api_key="test-key",
                timeout=60.0,
                connect_timeout=5.0,
            )
        call_args = mock_module.GrpcChannel.call_args
        assert call_args[0][5] == 60.0  # timeout
        assert call_args[0][6] == 5.0  # connect_timeout

    def test_init_insecure(self, mock_channel: MagicMock) -> None:
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
            GrpcIndex(
                host="test-index.svc.pinecone.io",
                api_key="test-key",
                secure=False,
            )
        call_args = mock_module.GrpcChannel.call_args
        assert call_args[0][0].startswith("http://")  # endpoint
        assert call_args[0][4] is False  # secure


class TestUpsert:
    """Tests for GrpcIndex.upsert()."""

    def test_upsert_basic(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 2}

        result = grpc_index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
            ],
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 2
        mock_channel.upsert.assert_called_once()

    def test_upsert_with_namespace(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert(
            vectors=[{"id": "v1", "values": [0.1]}],
            namespace="my-ns",
        )

        call_args = mock_channel.upsert.call_args
        assert call_args[0][1] == "my-ns"

    def test_upsert_with_vector_objects(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert(
            vectors=[Vector(id="v1", values=[0.1, 0.2])],
        )

        call_args = mock_channel.upsert.call_args
        vec_dicts = call_args[0][0]
        assert vec_dicts[0]["id"] == "v1"
        assert vec_dicts[0]["values"] == [0.1, 0.2]

    def test_upsert_with_tuples(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert(
            vectors=[("v1", [0.1, 0.2])],
        )

        call_args = mock_channel.upsert.call_args
        vec_dicts = call_args[0][0]
        assert vec_dicts[0]["id"] == "v1"

    def test_upsert_empty_namespace_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1]}])

        call_args = mock_channel.upsert.call_args
        assert call_args[0][1] is None  # empty string -> None for GrpcChannel


class TestQuery:
    """Tests for GrpcIndex.query()."""

    def test_query_by_vector(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.query.return_value = {
            "matches": [
                {"id": "v1", "score": 0.95, "values": [0.1, 0.2], "metadata": {"key": "val"}},
                {"id": "v2", "score": 0.85, "values": [], "metadata": None},
            ],
            "namespace": "test-ns",
            "usage": {"read_units": 5},
        }

        result = grpc_index.query(top_k=2, vector=[0.1, 0.2], namespace="test-ns")

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 2
        assert result.matches[0].id == "v1"
        assert result.matches[0].score == 0.95
        assert result.namespace == "test-ns"
        assert result.usage is not None
        assert result.usage.read_units == 5

    def test_query_by_id(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.query.return_value = {
            "matches": [{"id": "v2", "score": 0.9, "values": [], "metadata": None}],
            "namespace": "",
        }

        result = grpc_index.query(top_k=1, id="v1")

        assert len(result.matches) == 1
        mock_channel.query.assert_called_once_with(
            1,
            vector=None,
            id="v1",
            namespace=None,
            filter=None,
            include_values=False,
            include_metadata=False,
            sparse_vector=None,
            scan_factor=None,
            max_candidates=None,
        )

    def test_query_validates_top_k(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            grpc_index.query(top_k=0, vector=[0.1])

    def test_query_validates_both_vector_and_id(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="not both"):
            grpc_index.query(top_k=1, vector=[0.1], id="v1")

    def test_query_validates_neither_vector_nor_id(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="got neither"):
            grpc_index.query(top_k=1)

    def test_query_with_filter(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.query.return_value = {"matches": [], "namespace": ""}

        grpc_index.query(
            top_k=5,
            vector=[0.1],
            filter={"genre": {"$eq": "drama"}},
        )

        call_kwargs = mock_channel.query.call_args
        assert call_kwargs[1]["filter"] == {"genre": {"$eq": "drama"}}

    def test_query_no_usage(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.query.return_value = {"matches": [], "namespace": ""}

        result = grpc_index.query(top_k=1, vector=[0.1])
        assert result.usage is None

    def test_query_with_sparse_values(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.query.return_value = {
            "matches": [
                {
                    "id": "v1",
                    "score": 0.9,
                    "values": [],
                    "sparse_values": {"indices": [0, 2], "values": [0.5, 0.3]},
                    "metadata": None,
                }
            ],
            "namespace": "",
        }

        result = grpc_index.query(top_k=1, vector=[0.1])
        assert result.matches[0].sparse_values is not None
        assert result.matches[0].sparse_values.indices == [0, 2]
        assert result.matches[0].sparse_values.values == [0.5, 0.3]


class TestFetch:
    """Tests for GrpcIndex.fetch()."""

    def test_fetch_basic(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.fetch.return_value = {
            "vectors": {
                "v1": {"values": [0.1, 0.2], "metadata": {"key": "val"}},
                "v2": {"values": [0.3, 0.4], "metadata": None},
            },
            "namespace": "test-ns",
            "usage": {"read_units": 2},
        }

        result = grpc_index.fetch(ids=["v1", "v2"], namespace="test-ns")

        assert isinstance(result, FetchResponse)
        assert len(result.vectors) == 2
        assert result.vectors["v1"].id == "v1"
        assert result.vectors["v1"].values == [0.1, 0.2]
        assert result.vectors["v1"].metadata == {"key": "val"}
        assert result.namespace == "test-ns"
        assert result.usage is not None
        assert result.usage.read_units == 2

    def test_fetch_empty_ids_raises(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="ids must be a non-empty list"):
            grpc_index.fetch(ids=[])

    def test_fetch_missing_vectors(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.fetch.return_value = {
            "vectors": {},
            "namespace": "",
        }

        result = grpc_index.fetch(ids=["nonexistent"])
        assert len(result.vectors) == 0


class TestDelete:
    """Tests for GrpcIndex.delete()."""

    def test_delete_by_ids(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.delete.return_value = {}

        grpc_index.delete(ids=["v1", "v2"])

        mock_channel.delete.assert_called_once_with(
            ids=["v1", "v2"],
            delete_all=False,
            namespace=None,
            filter=None,
        )

    def test_delete_all(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.delete.return_value = {}

        grpc_index.delete(delete_all=True, namespace="my-ns")

        mock_channel.delete.assert_called_once_with(
            ids=None,
            delete_all=True,
            namespace="my-ns",
            filter=None,
        )

    def test_delete_by_filter(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.delete.return_value = {}

        grpc_index.delete(filter={"genre": {"$eq": "obsolete"}})

        mock_channel.delete.assert_called_once_with(
            ids=None,
            delete_all=False,
            namespace=None,
            filter={"genre": {"$eq": "obsolete"}},
        )

    def test_delete_no_mode_raises(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="Must specify one"):
            grpc_index.delete()

    def test_delete_multiple_modes_raises(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="Cannot combine"):
            grpc_index.delete(ids=["v1"], delete_all=True)


class TestUpdate:
    """Tests for GrpcIndex.update()."""

    def test_update_by_id(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.update.return_value = {"matched_records": 1}

        result = grpc_index.update(id="v1", values=[0.1, 0.2])

        assert isinstance(result, UpdateResponse)
        assert result.matched_records == 1

    def test_update_with_metadata(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.update.return_value = {}

        result = grpc_index.update(id="v1", set_metadata={"key": "new_val"})

        assert result.matched_records is None
        call_kwargs = mock_channel.update.call_args[1]
        assert call_kwargs["set_metadata"] == {"key": "new_val"}

    def test_update_with_sparse_values_model(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        from pinecone.models.vectors.sparse import SparseValues

        mock_channel.update.return_value = {}

        grpc_index.update(
            id="v1",
            sparse_values=SparseValues(indices=[0, 2], values=[0.5, 0.3]),
        )

        call_kwargs = mock_channel.update.call_args[1]
        assert call_kwargs["sparse_values"] == {"indices": [0, 2], "values": [0.5, 0.3]}

    def test_update_with_sparse_values_dict(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.update.return_value = {}

        sv_dict: dict[str, Any] = {"indices": [1, 3], "values": [0.7, 0.1]}
        grpc_index.update(id="v1", sparse_values=sv_dict)

        call_kwargs = mock_channel.update.call_args[1]
        assert call_kwargs["sparse_values"] == sv_dict

    def test_update_validates_both_id_and_filter(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="not both"):
            grpc_index.update(id="v1", filter={"key": "val"})

    def test_update_validates_neither_id_nor_filter(self, grpc_index: GrpcIndex) -> None:
        with pytest.raises(ValidationError, match="got neither"):
            grpc_index.update(values=[0.1])

    def test_update_by_filter_with_dry_run(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.update.return_value = {"matched_records": 42}

        result = grpc_index.update(
            filter={"genre": {"$eq": "drama"}},
            set_metadata={"year": 2020},
            dry_run=True,
        )

        assert result.matched_records == 42
        call_kwargs = mock_channel.update.call_args[1]
        assert call_kwargs["dry_run"] is True


class TestList:
    """Tests for GrpcIndex.list() and list_paginated()."""

    def test_list_paginated_basic(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.list.return_value = {
            "vectors": [{"id": "v1"}, {"id": "v2"}],
            "namespace": "test-ns",
            "usage": {"read_units": 1},
        }

        result = grpc_index.list_paginated(namespace="test-ns")

        assert isinstance(result, ListResponse)
        assert len(result.vectors) == 2
        assert result.vectors[0].id == "v1"
        assert result.namespace == "test-ns"
        assert result.usage is not None
        assert result.pagination is None

    def test_list_paginated_with_pagination(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.list.return_value = {
            "vectors": [{"id": "v1"}],
            "namespace": "",
            "pagination": {"next": "token123"},
        }

        result = grpc_index.list_paginated()

        assert result.pagination is not None
        assert result.pagination.next == "token123"

    def test_list_auto_paginates(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.list.side_effect = [
            {
                "vectors": [{"id": "v1"}],
                "namespace": "",
                "pagination": {"next": "page2"},
            },
            {
                "vectors": [{"id": "v2"}],
                "namespace": "",
            },
        ]

        pages = list(grpc_index.list())

        assert len(pages) == 2
        assert pages[0].vectors[0].id == "v1"
        assert pages[1].vectors[0].id == "v2"

    def test_list_stops_on_empty(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.list.return_value = {
            "vectors": [],
            "namespace": "",
        }

        pages = list(grpc_index.list())
        assert len(pages) == 0

    def test_list_returns_iterator(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.list.return_value = {
            "vectors": [{"id": "v1"}],
            "namespace": "",
        }

        result = grpc_index.list()
        assert isinstance(result, Iterator)


class TestDescribeIndexStats:
    """Tests for GrpcIndex.describe_index_stats()."""

    def test_describe_basic(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.describe_index_stats.return_value = {
            "namespaces": {
                "ns1": {"vector_count": 100},
                "ns2": {"vector_count": 200},
            },
            "dimension": 128,
            "index_fullness": 0.5,
            "total_vector_count": 300,
            "metric": "cosine",
            "vector_type": "dense",
            "memory_fullness": 0.3,
            "storage_fullness": 0.4,
        }

        result = grpc_index.describe_index_stats()

        assert isinstance(result, DescribeIndexStatsResponse)
        assert len(result.namespaces) == 2
        assert result.namespaces["ns1"].vector_count == 100
        assert result.namespaces["ns2"].vector_count == 200
        assert result.dimension == 128
        assert result.index_fullness == 0.5
        assert result.total_vector_count == 300
        assert result.metric == "cosine"
        assert result.vector_type == "dense"
        assert result.memory_fullness == 0.3
        assert result.storage_fullness == 0.4

    def test_describe_with_filter(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        mock_channel.describe_index_stats.return_value = {
            "namespaces": {},
            "dimension": 64,
            "index_fullness": 0.0,
            "total_vector_count": 0,
        }

        grpc_index.describe_index_stats(filter={"genre": {"$eq": "drama"}})

        mock_channel.describe_index_stats.assert_called_once_with(
            filter={"genre": {"$eq": "drama"}},
        )

    def test_describe_minimal_response(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.describe_index_stats.return_value = {
            "namespaces": {},
            "index_fullness": 0.0,
            "total_vector_count": 0,
        }

        result = grpc_index.describe_index_stats()

        assert result.dimension is None
        assert result.metric is None
        assert result.vector_type is None
        assert result.memory_fullness is None
        assert result.storage_fullness is None


class TestGrpcIndexClose:
    """Tests for GrpcIndex.close()."""

    def test_close_shuts_down_executor_before_channel(self, mock_channel: MagicMock) -> None:
        call_order: list[str] = []

        mock_executor = MagicMock()
        mock_executor.shutdown.side_effect = lambda wait: call_order.append(
            f"shutdown(wait={wait})"
        )
        mock_channel.close.side_effect = lambda: call_order.append("channel.close()")

        mock_http = MagicMock()
        mock_http.close.side_effect = lambda: call_order.append("http.close()")

        idx = _make_grpc_index(mock_channel)
        idx._executor = mock_executor
        idx._http = mock_http

        idx.close()

        assert call_order == ["shutdown(wait=True)", "http.close()", "channel.close()"]
        mock_executor.shutdown.assert_called_once_with(wait=True)
        mock_http.close.assert_called_once()


class TestGrpcErrorWrapping:
    """Tests for GrpcIndex._call_channel error wrapping."""

    def test_channel_exception_wrapped_as_connection_error(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.upsert.side_effect = RuntimeError("connection refused")

        with pytest.raises(PineconeConnectionError):
            grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1]}])

    def test_original_message_preserved(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        mock_channel.query.side_effect = RuntimeError("transport error: broken pipe")

        with pytest.raises(PineconeConnectionError) as exc_info:
            grpc_index.query(top_k=1, vector=[0.1])

        assert "transport error: broken pipe" in str(exc_info.value)

    def test_exception_chaining(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        original = RuntimeError("tls handshake failed")
        mock_channel.fetch.side_effect = original

        with pytest.raises(PineconeConnectionError) as exc_info:
            grpc_index.fetch(ids=["v1"])

        assert exc_info.value.__cause__ is original

    def test_invalid_argument_raises_api_error_400(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """gRPC INVALID_ARGUMENT maps to ApiError(status_code=400) for transport parity."""
        mock_channel.upsert.side_effect = PineconeValueError(
            "gRPC INVALID_ARGUMENT: Vector dimension 3 does not match the dimension of the index 2"
        )

        with pytest.raises(ApiError) as exc_info:
            grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2, 0.3]}])

        err = exc_info.value
        assert err.status_code == 400
        assert "INVALID_ARGUMENT" in str(err)

    def test_invalid_argument_exception_chained(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        original = PineconeValueError("gRPC INVALID_ARGUMENT: bad dimension")
        mock_channel.upsert.side_effect = original

        with pytest.raises(ApiError) as exc_info:
            grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1]}])

        assert exc_info.value.__cause__ is original

    def test_not_found_raises_not_found_error(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """gRPC NOT_FOUND maps to NotFoundError(status_code=404)."""
        mock_channel.fetch.side_effect = RuntimeError("gRPC NOT_FOUND: index does not exist")

        with pytest.raises(NotFoundError) as exc_info:
            grpc_index.fetch(ids=["v1"])

        assert exc_info.value.status_code == 404

    def test_unauthenticated_raises_unauthorized_error(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """gRPC UNAUTHENTICATED maps to UnauthorizedError(status_code=401)."""
        mock_channel.query.side_effect = RuntimeError("gRPC UNAUTHENTICATED: invalid api key")

        with pytest.raises(UnauthorizedError) as exc_info:
            grpc_index.query(top_k=1, vector=[0.1])

        assert exc_info.value.status_code == 401

    def test_permission_denied_raises_forbidden_error(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """gRPC PERMISSION_DENIED maps to ForbiddenError(status_code=403)."""
        mock_channel.describe_index_stats.side_effect = RuntimeError(
            "gRPC PERMISSION_DENIED: access denied"
        )

        with pytest.raises(ForbiddenError) as exc_info:
            grpc_index.describe_index_stats()

        assert exc_info.value.status_code == 403

    def test_unavailable_raises_connection_error(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """gRPC UNAVAILABLE (transport failure) remains PineconeConnectionError."""
        mock_channel.upsert.side_effect = RuntimeError("gRPC UNAVAILABLE: connection refused")

        with pytest.raises(PineconeConnectionError):
            grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1]}])

    def test_deadline_exceeded_raises_timeout_error(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """gRPC DEADLINE_EXCEEDED maps to PineconeTimeoutError for transport parity."""
        original = RuntimeError("gRPC DEADLINE_EXCEEDED: timeout")
        mock_channel.query.side_effect = original

        with pytest.raises(PineconeTimeoutError) as exc_info:
            grpc_index.query(top_k=1, vector=[0.1])

        assert "DEADLINE_EXCEEDED" in str(exc_info.value)
        assert exc_info.value.__cause__ is original

    def test_already_exists_raises_conflict_error(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """gRPC ALREADY_EXISTS maps to ConflictError(status_code=409) for transport parity."""
        original = RuntimeError("gRPC ALREADY_EXISTS: resource exists")
        mock_channel.upsert.side_effect = original

        with pytest.raises(ConflictError) as exc_info:
            grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1]}])

        assert exc_info.value.status_code == 409
        assert exc_info.value.__cause__ is original

    def test_internal_raises_service_error(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """gRPC INTERNAL maps to ServiceError(status_code=500) for transport parity."""
        original = RuntimeError("gRPC INTERNAL: server error")
        mock_channel.fetch.side_effect = original

        with pytest.raises(ServiceError) as exc_info:
            grpc_index.fetch(ids=["v1"])

        assert exc_info.value.status_code == 500
        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Records API — REST fallback in GrpcIndex
# ---------------------------------------------------------------------------

_INDEX_HOST = "test-index-abc123.svc.pinecone.io"
_INDEX_HOST_HTTPS = f"https://{_INDEX_HOST}"
_UPSERT_RECORDS_URL = f"{_INDEX_HOST_HTTPS}/records/namespaces/test-ns/upsert"
_SEARCH_URL = f"{_INDEX_HOST_HTTPS}/records/namespaces/test-ns/search"

_SEARCH_RESPONSE: dict[str, object] = {
    "result": {
        "hits": [
            {"_id": "doc-1", "_score": 0.95, "fields": {"text": "vector databases"}},
            {"_id": "doc-2", "_score": 0.80, "fields": {"text": "similarity search"}},
        ]
    },
    "usage": {"read_units": 3, "embed_total_tokens": 12},
}


class TestGrpcIndexUpsertRecords:
    """GrpcIndex.upsert_records() delegates to REST (NDJSON) endpoint."""

    @respx.mock
    def test_upsert_records_basic(self, mock_channel: MagicMock) -> None:
        respx.post(_UPSERT_RECORDS_URL).mock(return_value=httpx.Response(201))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)

        result = idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}, {"_id": "r2", "text": "world"}],
        )

        assert isinstance(result, UpsertRecordsResponse)
        assert result.record_count == 2

    @respx.mock
    def test_upsert_records_ndjson_content_type(self, mock_channel: MagicMock) -> None:
        route = respx.post(_UPSERT_RECORDS_URL).mock(return_value=httpx.Response(201))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.upsert_records(namespace="test-ns", records=[{"_id": "r1", "text": "hello"}])

        assert route.calls.last.request.headers["Content-Type"] == "application/x-ndjson"

    @respx.mock
    def test_upsert_records_ndjson_body(self, mock_channel: MagicMock) -> None:
        route = respx.post(_UPSERT_RECORDS_URL).mock(return_value=httpx.Response(201))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}, {"_id": "r2", "text": "world"}],
        )

        body = route.calls.last.request.content.decode("utf-8")
        lines = body.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["_id"] == "r1"
        assert json.loads(lines[1])["_id"] == "r2"

    @respx.mock
    def test_upsert_records_id_alias_normalized(self, mock_channel: MagicMock) -> None:
        """Records with 'id' key are normalized to '_id' in the NDJSON body."""
        route = respx.post(_UPSERT_RECORDS_URL).mock(return_value=httpx.Response(201))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.upsert_records(namespace="test-ns", records=[{"id": "r1", "text": "hello"}])

        body = route.calls.last.request.content.decode("utf-8")
        parsed = json.loads(body.strip())
        assert parsed["_id"] == "r1"
        assert "id" not in parsed

    def test_upsert_records_empty_namespace_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="non-empty"):
            idx.upsert_records(namespace="", records=[{"_id": "r1"}])

    def test_upsert_records_empty_records_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="non-empty"):
            idx.upsert_records(namespace="test-ns", records=[])

    def test_upsert_records_missing_id_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match=r"_id.*id"):
            idx.upsert_records(namespace="test-ns", records=[{"text": "no id here"}])


class TestGrpcIndexSearch:
    """GrpcIndex.search() delegates to REST endpoint."""

    @respx.mock
    def test_search_with_text_inputs(self, mock_channel: MagicMock) -> None:
        respx.post(_SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)

        response = idx.search(namespace="test-ns", top_k=5, inputs={"text": "vector search"})

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) == 2
        assert response.result.hits[0].id == "doc-1"
        assert response.result.hits[0].score == pytest.approx(0.95)
        assert response.usage.read_units == 3

    @respx.mock
    def test_search_request_body_inputs(self, mock_channel: MagicMock) -> None:
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.search(namespace="test-ns", top_k=5, inputs={"text": "vector search"})

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["top_k"] == 5
        assert body["query"]["inputs"] == {"text": "vector search"}

    @respx.mock
    def test_search_with_vector(self, mock_channel: MagicMock) -> None:
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.search(namespace="test-ns", top_k=5, vector=[0.1, 0.2, 0.3])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == [0.1, 0.2, 0.3]

    @respx.mock
    def test_search_with_rerank(self, mock_channel: MagicMock) -> None:
        route = respx.post(_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_SEARCH_RESPONSE)
        )
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        idx.search(
            namespace="test-ns",
            top_k=5,
            inputs={"text": "query"},
            rerank={"model": "bge-reranker-v2-m3", "rank_fields": ["text"], "top_n": 3},
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["rerank"]["model"] == "bge-reranker-v2-m3"

    def test_search_empty_namespace_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="non-empty"):
            idx.search(namespace="", top_k=5, inputs={"text": "query"})

    def test_search_top_k_zero_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="top_k"):
            idx.search(namespace="test-ns", top_k=0, inputs={"text": "query"})

    def test_search_no_query_source_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="inputs, vector, or id"):
            idx.search(namespace="test-ns", top_k=5)

    def test_search_rerank_missing_model_raises(self, mock_channel: MagicMock) -> None:
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)
        with pytest.raises(ValidationError, match="model"):
            idx.search(
                namespace="test-ns",
                top_k=5,
                inputs={"text": "q"},
                rerank={"rank_fields": ["text"]},
            )

    @respx.mock
    def test_search_records_alias(self, mock_channel: MagicMock) -> None:
        """search_records() is an alias for search() and produces the same result."""
        respx.post(_SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))
        idx = _make_grpc_index(mock_channel, host=_INDEX_HOST)

        response = idx.search_records(namespace="test-ns", top_k=5, inputs={"text": "query"})

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) == 2
