"""Unit tests for Index.upsert() batch_size and show_progress parameters."""

from __future__ import annotations

import sys

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.vectors.responses import UpsertResponse
from pinecone.models.vectors.vector import Vector

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
UPSERT_URL = f"{INDEX_HOST_HTTPS}/vectors/upsert"


def _make_upsert_response(*, upserted_count: int) -> dict[str, object]:
    return {"upsertedCount": upserted_count}


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


def _make_vectors(n: int) -> list[Vector]:
    return [Vector(id=f"v{i}", values=[float(i), float(i + 1)]) for i in range(n)]


class TestUpsertNoBatchSize:
    """batch_size=None (default) sends a single request."""

    @respx.mock
    def test_upsert_no_batch_size_sends_single_request(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=100))
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(100), batch_size=None)
        assert len(route.calls) == 1
        assert result.upserted_count == 100

    @respx.mock
    def test_upsert_default_batch_size_sends_single_request(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=50))
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(50))
        assert len(route.calls) == 1
        assert result.upserted_count == 50

    @respx.mock
    def test_upsert_response_info_preserved_when_not_batched(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_upsert_response(upserted_count=5),
                headers={"X-Pinecone-Request-Id": "req-abc123"},
            )
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(5))
        assert result.response_info is not None


class TestUpsertWithBatchSize:
    """batch_size=N splits vectors and sends one request per chunk."""

    @respx.mock
    def test_upsert_with_batch_size_sends_multiple_requests(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            side_effect=[
                httpx.Response(200, json=_make_upsert_response(upserted_count=100)),
                httpx.Response(200, json=_make_upsert_response(upserted_count=100)),
                httpx.Response(200, json=_make_upsert_response(upserted_count=50)),
            ]
        )
        idx = _make_index()
        idx.upsert(vectors=_make_vectors(250), batch_size=100, show_progress=False)
        assert len(route.calls) == 3

        import orjson

        first_body = orjson.loads(route.calls[0].request.content)
        second_body = orjson.loads(route.calls[1].request.content)
        third_body = orjson.loads(route.calls[2].request.content)
        assert len(first_body["vectors"]) == 100
        assert len(second_body["vectors"]) == 100
        assert len(third_body["vectors"]) == 50

    @respx.mock
    def test_upsert_with_batch_size_aggregates_response(self) -> None:
        respx.post(UPSERT_URL).mock(
            side_effect=[
                httpx.Response(200, json=_make_upsert_response(upserted_count=100)),
                httpx.Response(200, json=_make_upsert_response(upserted_count=100)),
                httpx.Response(200, json=_make_upsert_response(upserted_count=50)),
            ]
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(250), batch_size=100, show_progress=False)
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 250

    @respx.mock
    def test_upsert_response_info_is_none_when_batched(self) -> None:
        respx.post(UPSERT_URL).mock(
            side_effect=[
                httpx.Response(200, json=_make_upsert_response(upserted_count=100)),
                httpx.Response(200, json=_make_upsert_response(upserted_count=100)),
                httpx.Response(200, json=_make_upsert_response(upserted_count=50)),
            ]
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(250), batch_size=100, show_progress=False)
        assert result.response_info is None

    @respx.mock
    def test_upsert_namespace_forwarded_per_batch(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            side_effect=[
                httpx.Response(200, json=_make_upsert_response(upserted_count=5)),
                httpx.Response(200, json=_make_upsert_response(upserted_count=5)),
            ]
        )
        idx = _make_index()
        idx.upsert(vectors=_make_vectors(10), batch_size=5, namespace="my-ns", show_progress=False)

        import orjson

        for call in route.calls:
            body = orjson.loads(call.request.content)
            assert body.get("namespace") == "my-ns"

    @respx.mock
    def test_upsert_timeout_forwarded_per_batch(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            side_effect=[
                httpx.Response(200, json=_make_upsert_response(upserted_count=5)),
                httpx.Response(200, json=_make_upsert_response(upserted_count=5)),
            ]
        )
        idx = _make_index()
        idx.upsert(vectors=_make_vectors(10), batch_size=5, timeout=5.0, show_progress=False)
        assert len(route.calls) == 2

    @respx.mock
    def test_upsert_empty_vectors_with_batch_size(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json=_make_upsert_response(upserted_count=0))
        )
        idx = _make_index()
        result = idx.upsert(vectors=[], batch_size=100, show_progress=False)
        assert len(route.calls) == 0
        assert result.upserted_count == 0


class TestUpsertInvalidBatchSize:
    """Invalid batch_size values raise PineconeValueError."""

    def test_upsert_invalid_batch_size_zero(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError):
            idx.upsert(vectors=_make_vectors(5), batch_size=0)

    def test_upsert_invalid_batch_size_negative(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError):
            idx.upsert(vectors=_make_vectors(5), batch_size=-1)

    def test_upsert_invalid_batch_size_float(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError):
            idx.upsert(vectors=_make_vectors(5), batch_size=1.5)  # type: ignore[arg-type]


class TestUpsertShowProgress:
    """show_progress behavior with and without tqdm."""

    @respx.mock
    def test_upsert_show_progress_false_does_not_import_tqdm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "tqdm", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)  # type: ignore[arg-type]
        respx.post(UPSERT_URL).mock(
            side_effect=[
                httpx.Response(200, json=_make_upsert_response(upserted_count=5)),
                httpx.Response(200, json=_make_upsert_response(upserted_count=5)),
            ]
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(10), batch_size=5, show_progress=False)
        assert result.upserted_count == 10

    @respx.mock
    def test_upsert_show_progress_true_works_without_tqdm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "tqdm", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)  # type: ignore[arg-type]
        respx.post(UPSERT_URL).mock(
            side_effect=[
                httpx.Response(200, json=_make_upsert_response(upserted_count=5)),
                httpx.Response(200, json=_make_upsert_response(upserted_count=5)),
            ]
        )
        idx = _make_index()
        result = idx.upsert(vectors=_make_vectors(10), batch_size=5, show_progress=True)
        assert result.upserted_count == 10
