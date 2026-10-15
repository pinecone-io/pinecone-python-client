"""``top_k`` and ``limit`` on ``index.documents`` are type-checked client-side (#707).

A non-``int`` (``bool`` included) raises ``PineconeTypeError`` naming the parameter
and sends no request, on the sync and async twins alike. The type check is its own
step, separate from the range check.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import PineconeTypeError, PineconeValueError
from pinecone.index import Index
from pinecone.models.documents.requests import (
    FetchDocumentsRequest,
    ListDocumentsRequest,
    SearchDocumentsRequest,
    _NotAnIntegerError,
)
from pinecone.models.documents.score_by import TextQuery

INDEX_HOST = "documents-index-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
NS = "articles-en"

BAD_VALUES = ["3", 3.0, True]
BAD_IDS = ["str", "float", "bool"]
QUERY = [TextQuery(query="dark matter", fields=["abstract"])]


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=INDEX_HOST, api_key="test-key")
    yield client
    client.close()


@pytest.fixture
async def aindex() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
    yield client
    await client.close()


def _mock_all() -> respx.Route:
    return respx.route().mock(return_value=httpx.Response(500))


class TestSyncDocuments:
    @respx.mock
    @pytest.mark.parametrize("value", BAD_VALUES, ids=BAD_IDS)
    def test_search_top_k(self, index: Index, value: Any) -> None:
        route = _mock_all()
        with pytest.raises(PineconeTypeError, match="top_k"):
            index.documents.search(namespace=NS, score_by=QUERY, top_k=value)
        assert route.call_count == 0

    @respx.mock
    @pytest.mark.parametrize("value", BAD_VALUES, ids=BAD_IDS)
    def test_fetch_limit(self, index: Index, value: Any) -> None:
        route = _mock_all()
        with pytest.raises(PineconeTypeError, match="limit"):
            index.documents.fetch(namespace=NS, filter={"a": {"$eq": 1}}, limit=value)
        assert route.call_count == 0

    @respx.mock
    @pytest.mark.parametrize("value", BAD_VALUES, ids=BAD_IDS)
    def test_list_limit(self, index: Index, value: Any) -> None:
        route = _mock_all()
        with pytest.raises(PineconeTypeError, match="limit"):
            index.documents.list(namespace=NS, limit=value).to_list()
        assert route.call_count == 0


class TestAsyncDocuments:
    @respx.mock
    @pytest.mark.parametrize("value", BAD_VALUES, ids=BAD_IDS)
    async def test_search_top_k(self, aindex: AsyncIndex, value: Any) -> None:
        route = _mock_all()
        with pytest.raises(PineconeTypeError, match="top_k"):
            await aindex.documents.search(namespace=NS, score_by=QUERY, top_k=value)
        assert route.call_count == 0

    @respx.mock
    @pytest.mark.parametrize("value", BAD_VALUES, ids=BAD_IDS)
    async def test_fetch_limit(self, aindex: AsyncIndex, value: Any) -> None:
        route = _mock_all()
        with pytest.raises(PineconeTypeError, match="limit"):
            await aindex.documents.fetch(namespace=NS, filter={"a": {"$eq": 1}}, limit=value)
        assert route.call_count == 0

    @respx.mock
    @pytest.mark.parametrize("value", BAD_VALUES, ids=BAD_IDS)
    async def test_list_limit(self, aindex: AsyncIndex, value: Any) -> None:
        route = _mock_all()
        with pytest.raises(PineconeTypeError, match="limit"):
            await aindex.documents.list(namespace=NS, limit=value).to_list()
        assert route.call_count == 0


class TestTypeCheckIsIndependentOfRangeCheck:
    """The int check fires on values the range check would also have to compare."""

    @pytest.mark.parametrize("value", BAD_VALUES, ids=BAD_IDS)
    def test_message_is_about_type_not_range(self, value: Any) -> None:
        for build in (
            lambda: SearchDocumentsRequest(
                score_by=[{"type": "text", "fields": ["a"], "query": "q"}], top_k=value
            ),
            lambda: FetchDocumentsRequest(filter={"a": {"$eq": 1}}, limit=value),
            lambda: ListDocumentsRequest(limit=value),
        ):
            with pytest.raises(_NotAnIntegerError, match="must be an integer"):
                build()

    def test_valid_ints_still_accepted(self) -> None:
        SearchDocumentsRequest(score_by=[{"type": "text", "fields": ["a"], "query": "q"}], top_k=10)
        FetchDocumentsRequest(filter={"a": {"$eq": 1}}, limit=10)
        ListDocumentsRequest(limit=None)


class TestTypeErrorContract:
    """The wrong-type rejection is a ``TypeError``, never a ``PineconeValueError``."""

    @respx.mock
    @pytest.mark.parametrize("value", BAD_VALUES, ids=BAD_IDS)
    def test_caught_by_except_type_error_and_not_value_error(
        self, index: Index, value: Any
    ) -> None:
        route = _mock_all()
        with pytest.raises(PineconeTypeError) as excinfo:
            index.documents.search(namespace=NS, score_by=QUERY, top_k=value)
        assert isinstance(excinfo.value, TypeError)
        assert not isinstance(excinfo.value, PineconeValueError)
        assert route.call_count == 0

    @respx.mock
    def test_range_error_stays_value_error(self, index: Index) -> None:
        route = _mock_all()
        with pytest.raises(PineconeValueError, match="top_k"):
            index.documents.search(namespace=NS, score_by=QUERY, top_k=0)
        assert route.call_count == 0
