"""LSN properties read the headers the documents API actually sends.

The server reports a write's log position in ``x-pinecone-request-lsn`` and a
search's catch-up position in ``x-pinecone-max-indexed-lsn``. The vectors-API
names (``x-pinecone-lsn-committed`` / ``x-pinecone-lsn-reconciled``) remain as
fallbacks because the vectors bulk path synthesizes them.
"""

from __future__ import annotations

import textwrap
import time
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace

import httpx
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.index import Index
from pinecone.models.documents.score_by import TextQuery
from pinecone.models.response_info import ResponseInfo

HOST = "documents-index-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{HOST}"
NS = "articles-en"
WRITE_HEADERS = {"x-pinecone-request-lsn": "28"}
SEARCH_HEADERS = {"x-pinecone-max-indexed-lsn": "27"}
SEARCH_BODY = {"matches": [], "namespace": NS, "usage": {"read_units": 1}}
FETCH_BODY = {"documents": {}, "namespace": NS, "usage": {"read_units": 1}}


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=HOST, api_key="test-key")
    yield client
    client.close()


@pytest.fixture
async def aindex() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=HOST, api_key="test-key")
    yield client
    await client.close()


class TestResponseInfoHeaderNames:
    def test_committed_reads_request_lsn(self) -> None:
        assert ResponseInfo(raw_headers=WRITE_HEADERS).lsn_committed == 28

    def test_reconciled_reads_max_indexed_lsn(self) -> None:
        assert ResponseInfo(raw_headers=SEARCH_HEADERS).lsn_reconciled == 27

    def test_vectors_api_names_remain_as_fallbacks(self) -> None:
        info = ResponseInfo(
            raw_headers={"x-pinecone-lsn-committed": "5", "x-pinecone-lsn-reconciled": "4"}
        )
        assert info.lsn_committed == 5
        assert info.lsn_reconciled == 4

    def test_documents_names_win_when_both_are_present(self) -> None:
        info = ResponseInfo(
            raw_headers={
                "x-pinecone-request-lsn": "28",
                "x-pinecone-lsn-committed": "5",
                "x-pinecone-max-indexed-lsn": "27",
                "x-pinecone-lsn-reconciled": "4",
            }
        )
        assert info.lsn_committed == 28
        assert info.lsn_reconciled == 27

    def test_unparseable_documents_header_falls_back(self) -> None:
        info = ResponseInfo(
            raw_headers={"x-pinecone-request-lsn": "n/a", "x-pinecone-lsn-committed": "5"}
        )
        assert info.lsn_committed == 5

    def test_is_reconciled_uses_search_header(self) -> None:
        info = ResponseInfo(raw_headers=SEARCH_HEADERS)
        assert info.is_reconciled(27)
        assert not info.is_reconciled(28)

    def test_fetch_style_headers_leave_reconciled_unknown(self) -> None:
        info = ResponseInfo(raw_headers={})
        assert info.lsn_reconciled is None
        assert not info.is_reconciled(1)

    def test_recipe_polls_search_not_fetch(self) -> None:
        doc = ResponseInfo.__doc__ or ""
        recipe = doc[doc.index("Examples:") :]
        assert "documents.search(" in recipe
        assert "documents.fetch(" not in recipe
        assert "fetch" in doc and "no reconciled position" in doc

    def test_request_id_doc_says_documents_api_sends_none(self) -> None:
        assert "Documents-API responses do not carry one today" in (
            ResponseInfo.request_id.__doc__ or ""
        )


def _recipe() -> str:
    doc = ResponseInfo.__doc__ or ""
    block = doc.split(".. code-block:: python", 1)[1].split(".. versionchanged::", 1)[0]
    return textwrap.dedent(block)


def _run_recipe(index: Index, reconciled: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    """Execute the class docstring's example; return how many searches it made."""
    clock = iter(range(0, 10_000, 10))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(f"{BASE_URL}/namespaces/published/documents/upsert").mock(
        return_value=httpx.Response(202, json={"upserted_count": 1}, headers=WRITE_HEADERS)
    )
    positions = iter(reconciled)
    search = respx.post(f"{BASE_URL}/namespaces/published/documents/search").mock(
        side_effect=lambda request: httpx.Response(
            200, json=SEARCH_BODY, headers={"x-pinecone-max-indexed-lsn": next(positions)}
        )
    )
    exec(_recipe(), {"pc": SimpleNamespace(index=lambda name: index)})  # noqa: S102
    return search.call_count


class TestDocstringRecipe:
    @respx.mock
    def test_recipe_stops_once_search_reaches_the_write(
        self, index: Index, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _run_recipe(index, ["26", "27", "28"], monkeypatch) == 3

    @respx.mock
    def test_recipe_gives_up_at_its_deadline(
        self, index: Index, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(TimeoutError, match="LSN 28"):
            _run_recipe(index, ["27"] * 100, monkeypatch)


class TestSyncDocumentsResponseInfo:
    @respx.mock
    def test_upsert_update_delete_report_committed(self, index: Index) -> None:
        for op in ("upsert", "update", "delete"):
            respx.post(f"{BASE_URL}/namespaces/{NS}/documents/{op}").mock(
                return_value=httpx.Response(
                    202, json={"upserted_count": 1} if op == "upsert" else {}, headers=WRITE_HEADERS
                )
            )
        up = index.documents.upsert(namespace=NS, documents=[{"_id": "a"}])
        upd = index.documents.update(namespace=NS, documents=[{"_id": "a", "t": 1}])
        dele = index.documents.delete(namespace=NS, ids=["a"])
        for resp in (up, upd, dele):
            assert resp.response_info is not None
            assert resp.response_info.lsn_committed == 28

    @respx.mock
    def test_search_reports_reconciled(self, index: Index) -> None:
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/search").mock(
            return_value=httpx.Response(200, json=SEARCH_BODY, headers=SEARCH_HEADERS)
        )
        result = index.documents.search(
            namespace=NS, top_k=1, score_by=[TextQuery(query="t", fields=["title"])]
        )
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 27

    @respx.mock
    def test_fetch_carries_no_reconciled_position(self, index: Index) -> None:
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/fetch").mock(
            return_value=httpx.Response(200, json=FETCH_BODY)
        )
        result = index.documents.fetch(namespace=NS, ids=["a"])
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled is None

    @respx.mock
    def test_batch_upsert_response_info_carries_highest_committed(self, index: Index) -> None:
        lsns = iter(["28", "31", "30"])
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            side_effect=lambda request: httpx.Response(
                202,
                json={"upserted_count": 2},
                headers={"x-pinecone-request-lsn": next(lsns)},
            )
        )
        result = index.documents.batch_upsert(
            namespace=NS,
            documents=[{"_id": f"d{i}"} for i in range(6)],
            batch_size=2,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is not None
        assert result.response_info.lsn_committed == 31


class TestAsyncDocumentsResponseInfo:
    @respx.mock
    async def test_upsert_update_delete_report_committed(self, aindex: AsyncIndex) -> None:
        for op in ("upsert", "update", "delete"):
            respx.post(f"{BASE_URL}/namespaces/{NS}/documents/{op}").mock(
                return_value=httpx.Response(
                    202, json={"upserted_count": 1} if op == "upsert" else {}, headers=WRITE_HEADERS
                )
            )
        up = await aindex.documents.upsert(namespace=NS, documents=[{"_id": "a"}])
        upd = await aindex.documents.update(namespace=NS, documents=[{"_id": "a", "t": 1}])
        dele = await aindex.documents.delete(namespace=NS, ids=["a"])
        for resp in (up, upd, dele):
            assert resp.response_info is not None
            assert resp.response_info.lsn_committed == 28

    @respx.mock
    async def test_search_reports_reconciled(self, aindex: AsyncIndex) -> None:
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/search").mock(
            return_value=httpx.Response(200, json=SEARCH_BODY, headers=SEARCH_HEADERS)
        )
        result = await aindex.documents.search(
            namespace=NS, top_k=1, score_by=[TextQuery(query="t", fields=["title"])]
        )
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled == 27

    @respx.mock
    async def test_fetch_carries_no_reconciled_position(self, aindex: AsyncIndex) -> None:
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/fetch").mock(
            return_value=httpx.Response(200, json=FETCH_BODY)
        )
        result = await aindex.documents.fetch(namespace=NS, ids=["a"])
        assert result.response_info is not None
        assert result.response_info.lsn_reconciled is None

    @respx.mock
    async def test_batch_upsert_response_info_carries_highest_committed(
        self, aindex: AsyncIndex
    ) -> None:
        lsns = iter(["28", "31", "30"])
        respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            side_effect=lambda request: httpx.Response(
                202,
                json={"upserted_count": 2},
                headers={"x-pinecone-request-lsn": next(lsns)},
            )
        )
        result = await aindex.documents.batch_upsert(
            namespace=NS,
            documents=[{"_id": f"d{i}"} for i in range(6)],
            batch_size=2,
            max_concurrency=1,
            show_progress=False,
        )
        assert result.response_info is not None
        assert result.response_info.lsn_committed == 31
