"""Unit tests for PreviewDocument response models and adapters."""

from __future__ import annotations

import httpx
import orjson

from pinecone.models.response_info import ResponseInfo
from pinecone.preview._internal.adapters.documents import PreviewDocumentsAdapter
from pinecone.preview.models.documents import (
    PreviewDocument,
    PreviewDocumentFetchResponse,
    PreviewDocumentSearchResponse,
    PreviewDocumentUpsertResponse,
    PreviewUsage,
)


def test_upsert_response_carries_upserted_count() -> None:
    r = PreviewDocumentUpsertResponse(upserted_count=3)
    assert r.upserted_count == 3


def test_decode_search_response_populates_matches_namespace_and_usage() -> None:
    payload = orjson.dumps(
        {
            "matches": [
                {"_id": "doc-1", "_score": 0.9, "title": "Rome"},
                {"_id": "doc-2", "_score": 0.7, "title": "Athens"},
            ],
            "namespace": "wiki",
            "usage": {"read_units": 5},
        }
    )
    response = PreviewDocumentsAdapter.to_search_response(httpx.Response(200, content=payload))
    assert response.namespace == "wiki"
    assert response.usage is not None
    assert response.usage.read_units == 5
    assert len(response.matches) == 2
    assert response.matches[0].id == "doc-1"
    assert response.matches[0].score == 0.9
    assert response.matches[1].id == "doc-2"
    assert response.matches[1].score == 0.7


def test_decode_search_response_usage_defaults_to_none_when_absent() -> None:
    payload = orjson.dumps(
        {
            "matches": [{"_id": "doc-1", "_score": 0.5}],
            "namespace": "ns",
        }
    )
    response = PreviewDocumentsAdapter.to_search_response(httpx.Response(200, content=payload))
    assert response.usage is None


def test_decode_search_response_dynamic_field_access_on_match() -> None:
    payload = orjson.dumps(
        {
            "matches": [{"_id": "doc-1", "_score": 0.8, "title": "Ancient Rome"}],
            "namespace": "ns",
        }
    )
    response = PreviewDocumentsAdapter.to_search_response(httpx.Response(200, content=payload))
    match = response.matches[0]
    assert match.title == "Ancient Rome"  # type: ignore[attr-defined]
    assert match.get("missing") is None


def test_decode_fetch_response_populates_documents_keyed_by_id() -> None:
    payload = orjson.dumps(
        {
            "documents": {
                "doc-1": {"_id": "doc-1", "title": "Rome"},
                "doc-2": {"_id": "doc-2", "title": "Athens"},
            },
            "namespace": "wiki",
            "usage": {"read_units": 2},
        }
    )
    response = PreviewDocumentsAdapter.to_fetch_response(httpx.Response(200, content=payload))
    assert set(response.documents.keys()) == {"doc-1", "doc-2"}
    assert isinstance(response.documents["doc-1"], PreviewDocument)
    assert response.documents["doc-1"].id == "doc-1"
    assert response.documents["doc-2"].id == "doc-2"


def test_decode_fetch_response_omits_missing_ids_from_map() -> None:
    # The API returns only the IDs that exist; absent IDs are not in the response.
    payload = orjson.dumps(
        {
            "documents": {
                "doc-1": {"_id": "doc-1"},
            },
            "namespace": "ns",
        }
    )
    response = PreviewDocumentsAdapter.to_fetch_response(httpx.Response(200, content=payload))
    assert "doc-99999" not in response.documents
    assert "doc-1" in response.documents


def test_decode_fetch_response_usage_defaults_to_none_when_absent() -> None:
    payload = orjson.dumps(
        {
            "documents": {"doc-1": {"_id": "doc-1"}},
            "namespace": "ns",
        }
    )
    response = PreviewDocumentsAdapter.to_fetch_response(httpx.Response(200, content=payload))
    assert response.usage is None


def test_search_response_repr_matches_spec_format() -> None:
    matches = [PreviewDocument({"_id": "doc-1", "_score": 0.9})]
    usage = PreviewUsage(read_units=3)
    r = PreviewDocumentSearchResponse(matches=matches, namespace="wiki", usage=usage)
    text = repr(r)
    assert text.startswith("SearchResponse(matches=")
    assert "namespace=" in text
    assert "usage=" in text


def test_search_response_repr_html_returns_html_string() -> None:
    matches = [PreviewDocument({"_id": "doc-1", "_score": 0.9})]
    r = PreviewDocumentSearchResponse(matches=matches, namespace="ns")
    html = r._repr_html_()
    assert isinstance(html, str)
    assert "<table" in html


def test_search_response_carries_response_info() -> None:
    matches = [PreviewDocument({"_id": "doc-1", "_score": 0.9})]
    info = ResponseInfo(request_id="req-1", lsn_reconciled=42, lsn_committed=50)
    r = PreviewDocumentSearchResponse(matches=matches, namespace="ns", response_info=info)
    assert r.response_info is not None
    assert r.response_info.lsn_reconciled == 42
    assert r.response_info.lsn_committed == 50
    assert r.response_info.request_id == "req-1"


def test_search_response_response_info_defaults_to_none() -> None:
    r = PreviewDocumentSearchResponse(matches=[], namespace="ns")
    assert r.response_info is None


def test_fetch_response_carries_response_info() -> None:
    info = ResponseInfo(request_id="req-1", lsn_reconciled=7, lsn_committed=9)
    r = PreviewDocumentFetchResponse(documents={}, namespace="ns", response_info=info)
    assert r.response_info is not None
    assert r.response_info.lsn_reconciled == 7


def test_fetch_response_response_info_defaults_to_none() -> None:
    r = PreviewDocumentFetchResponse(documents={}, namespace="ns")
    assert r.response_info is None
