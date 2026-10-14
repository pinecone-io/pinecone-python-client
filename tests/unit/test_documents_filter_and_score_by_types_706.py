"""A bare score clause or a filter ``Condition`` raises PineconeTypeError naming the fix (#706)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone import Field, PineconeTypeError, TextQuery
from pinecone.async_client.async_index import AsyncIndex
from pinecone.index import Index

HOST = "type-errors-abc123.svc.us-east-1-aws.pinecone.io"
CLAUSE = TextQuery(query="dark matter", fields=["abstract"])
CONDITION = (Field("category") == "cs") & Field("year").gte(2010)


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


def _calls(op: str) -> dict[str, Any]:
    ns = "arxiv"
    if op == "search":
        return {"namespace": ns, "top_k": 3, "score_by": [CLAUSE], "filter": CONDITION}
    if op == "fetch":
        return {"namespace": ns, "filter": CONDITION}
    if op == "delete":
        return {"namespace": ns, "filter": CONDITION}
    return {"namespace": ns, "filter": CONDITION, "set_fields": {"a": 1}}


OPS = ["search", "fetch", "delete", "update"]


def _gen() -> Iterator[TextQuery]:
    yield CLAUSE


SEARCH_OK = httpx.Response(
    200, json={"matches": [], "namespace": "arxiv", "usage": {"read_units": 1}}
)


class TestSync:
    @pytest.mark.parametrize("op", OPS)
    def test_condition_filter_names_parameter_and_to_dict(self, index: Index, op: str) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().respond(500)
            with pytest.raises(PineconeTypeError, match=r"filter.*to_dict\(\)"):
                getattr(index.documents, op)(**_calls(op))
            assert not route.called

    @pytest.mark.parametrize("bad", [CLAUSE, "text", {"text": {}}])
    def test_non_list_score_by_names_parameter_and_list(self, index: Index, bad: Any) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().respond(500)
            with pytest.raises(PineconeTypeError, match=r"score_by.*list"):
                index.documents.search(namespace="arxiv", top_k=3, score_by=bad)
            assert not route.called

    @pytest.mark.parametrize("make", [lambda: (CLAUSE,), _gen])
    def test_non_list_iterable_score_by_still_accepted(self, index: Index, make: Any) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=SEARCH_OK)
            index.documents.search(
                namespace="arxiv", top_k=3, score_by=make(), filter=CONDITION.to_dict()
            )
            assert route.called


class TestAsync:
    @pytest.mark.parametrize("op", OPS)
    async def test_condition_filter_names_parameter_and_to_dict(
        self, aindex: AsyncIndex, op: str
    ) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().respond(500)
            with pytest.raises(PineconeTypeError, match=r"filter.*to_dict\(\)"):
                await getattr(aindex.documents, op)(**_calls(op))
            assert not route.called

    @pytest.mark.parametrize("bad", [CLAUSE, "text", {"text": {}}])
    async def test_non_list_score_by_names_parameter_and_list(
        self, aindex: AsyncIndex, bad: Any
    ) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().respond(500)
            with pytest.raises(PineconeTypeError, match=r"score_by.*list"):
                await aindex.documents.search(namespace="arxiv", top_k=3, score_by=bad)
            assert not route.called

    @pytest.mark.parametrize("make", [lambda: (CLAUSE,), _gen])
    async def test_non_list_iterable_score_by_still_accepted(
        self, aindex: AsyncIndex, make: Any
    ) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=SEARCH_OK)
            await aindex.documents.search(
                namespace="arxiv", top_k=3, score_by=make(), filter=CONDITION.to_dict()
            )
            assert route.called
