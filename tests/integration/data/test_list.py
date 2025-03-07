import logging
import pytest
from ..helpers import poll_fetch_for_ids_in_namespace, embedding_values, random_string
import time

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def list_namespace():
    return random_string(10)


@pytest.fixture(scope="session")
def seed_for_list(idx, list_namespace, wait=True):
    logger.debug(f"Upserting into list namespace '{list_namespace}'")
    for i in range(0, 1000, 50):
        idx.upsert(
            vectors=[(str(i + d), embedding_values(2)) for d in range(50)], namespace=list_namespace
        )

    if wait:
        poll_fetch_for_ids_in_namespace(idx, ids=["999"], namespace=list_namespace)

    logger.debug(
        "Sleeping for 30 seconds to account for freshness being turned off for internal project index"
    )
    time.sleep(30)

    yield


@pytest.mark.usefixtures("seed_for_list")
class TestListPaginated:
    def test_list_when_no_results(self, idx):
        results = idx.list_paginated(namespace="no-results")
        assert results is not None
        assert results.namespace == "no-results"
        assert len(results.vectors) == 0
        # assert results.pagination == None

    def test_list_no_args(self, idx):
        results = idx.list_paginated()

        assert results is not None
        assert results.namespace == ""
        assert results.vectors is not None
        # assert results.pagination == None

    def test_list_when_limit(self, idx, list_namespace):
        results = idx.list_paginated(limit=10, namespace=list_namespace)

        assert results is not None
        assert len(results.vectors) == 10
        assert results.namespace == list_namespace
        assert results.pagination is not None
        assert results.pagination.next is not None
        assert isinstance(results.pagination.next, str)
        assert results.pagination.next != ""

    def test_list_when_using_pagination(self, idx, list_namespace):
        results = idx.list_paginated(prefix="99", limit=5, namespace=list_namespace)
        next_results = idx.list_paginated(
            prefix="99", limit=5, namespace=list_namespace, pagination_token=results.pagination.next
        )
        next_next_results = idx.list_paginated(
            prefix="99",
            limit=5,
            namespace=list_namespace,
            pagination_token=next_results.pagination.next,
        )

        assert results.namespace == list_namespace
        assert len(results.vectors) == 5
        assert [v.id for v in results.vectors] == ["99", "990", "991", "992", "993"]
        assert len(next_results.vectors) == 5
        assert [v.id for v in next_results.vectors] == ["994", "995", "996", "997", "998"]
        assert len(next_next_results.vectors) == 1
        assert [v.id for v in next_next_results.vectors] == ["999"]
        # assert next_next_results.pagination == None


@pytest.mark.usefixtures("seed_for_list")
class TestList:
    def test_list(self, idx, list_namespace):
        results = idx.list(prefix="99", limit=20, namespace=list_namespace)

        page_count = 0
        for ids in results:
            page_count += 1
            assert ids is not None
            assert len(ids) == 11
            assert ids == [
                "99",
                "990",
                "991",
                "992",
                "993",
                "994",
                "995",
                "996",
                "997",
                "998",
                "999",
            ]
        assert page_count == 1

    def test_list_when_no_results_for_prefix(self, idx, list_namespace):
        page_count = 0
        for ids in idx.list(prefix="no-results", namespace=list_namespace):
            page_count += 1
        assert page_count == 0

    def test_list_when_no_results_for_namespace(self, idx):
        page_count = 0
        for ids in idx.list(prefix="99", namespace="no-results"):
            page_count += 1
        assert page_count == 0

    def test_list_when_multiple_pages(self, idx, list_namespace):
        pages = []
        page_sizes = []
        page_count = 0

        for ids in idx.list(prefix="99", limit=5, namespace=list_namespace):
            page_count += 1
            assert ids is not None
            page_sizes.append(len(ids))
            pages.append(ids)

        assert page_count == 3
        assert page_sizes == [5, 5, 1]
        assert pages[0] == ["99", "990", "991", "992", "993"]
        assert pages[1] == ["994", "995", "996", "997", "998"]
        assert pages[2] == ["999"]

    def test_list_then_fetch(self, idx, list_namespace):
        vectors = []

        for ids in idx.list(prefix="99", limit=5, namespace=list_namespace):
            result = idx.fetch(ids=ids, namespace=list_namespace)
            vectors.extend([v for _, v in result.vectors.items()])

        assert len(vectors) == 11
        assert set([v.id for v in vectors]) == set(
            ["99", "990", "991", "992", "993", "994", "995", "996", "997", "998", "999"]
        )
