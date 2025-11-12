import logging
import time
import pytest
from tests.integration.helpers import embedding_values, random_string, poll_until_lsn_reconciled

logger = logging.getLogger(__name__)


def poll_until_list_has_results(
    idx, prefix: str, namespace: str, expected_count: int, max_wait_time: int = 120
):
    """Poll until list returns the expected number of results for a given prefix.

    Args:
        idx: The index client
        prefix: The prefix to search for
        namespace: The namespace to search in
        expected_count: The expected number of results
        max_wait_time: Maximum time to wait in seconds

    Raises:
        TimeoutError: If the expected count is not reached within max_wait_time seconds
    """
    time_waited = 0
    wait_per_iteration = 2
    last_count = None

    while time_waited < max_wait_time:
        # Try to list vectors with the prefix
        try:
            results = list(idx.list(prefix=prefix, namespace=namespace))
            total_count = sum(len(page) for page in results)
        except Exception as e:
            logger.warning(
                f"Error listing vectors with prefix '{prefix}' in namespace '{namespace}': {e}"
            )
            total_count = 0

        if total_count >= expected_count:
            logger.debug(
                f"List returned {total_count} results for prefix '{prefix}' in namespace '{namespace}'"
            )
            return

        # Log progress, including namespace stats if available
        if total_count != last_count:
            try:
                namespace_desc = idx.describe_namespace(namespace=namespace)
                record_count = (
                    int(namespace_desc.record_count)
                    if namespace_desc.record_count is not None
                    else 0
                )
                logger.debug(
                    f"Polling for list results. Prefix: '{prefix}', namespace: '{namespace}', "
                    f"current count: {total_count}, expected: {expected_count}, "
                    f"namespace record_count: {record_count}, waited: {time_waited}s"
                )
            except Exception:
                logger.debug(
                    f"Polling for list results. Prefix: '{prefix}', namespace: '{namespace}', "
                    f"current count: {total_count}, expected: {expected_count}, waited: {time_waited}s"
                )
            last_count = total_count

        time.sleep(wait_per_iteration)
        time_waited += wait_per_iteration

    # On timeout, provide more diagnostic information
    try:
        namespace_desc = idx.describe_namespace(namespace=namespace)
        record_count = (
            int(namespace_desc.record_count) if namespace_desc.record_count is not None else 0
        )
        final_results = list(idx.list(prefix=prefix, namespace=namespace))
        final_count = sum(len(page) for page in final_results)
        raise TimeoutError(
            f"Timeout waiting for list to return {expected_count} results for prefix '{prefix}' "
            f"in namespace '{namespace}' after {time_waited} seconds. "
            f"Final count: {final_count}, namespace record_count: {record_count}"
        )
    except Exception as e:
        if isinstance(e, TimeoutError):
            raise
        raise TimeoutError(
            f"Timeout waiting for list to return {expected_count} results for prefix '{prefix}' "
            f"in namespace '{namespace}' after {time_waited} seconds. "
            f"Error getting diagnostics: {e}"
        )


@pytest.fixture(scope="session")
def list_namespace():
    return random_string(10)


def poll_namespace_until_ready(idx, namespace: str, expected_count: int, max_wait_time: int = 60):
    """Poll describe_namespace until it has the expected record count.

    Args:
        idx: The index client
        namespace: The namespace to check
        expected_count: The expected record count
        max_wait_time: Maximum time to wait in seconds

    Raises:
        TimeoutError: If the expected count is not reached within max_wait_time seconds
        NotFoundException: If the namespace doesn't exist after waiting (this is expected in some tests)
    """
    from pinecone.exceptions import NotFoundException

    time_waited = 0
    wait_per_iteration = 2
    not_found_count = 0
    max_not_found_retries = 19  # Allow a few NotFoundExceptions before giving up

    while time_waited < max_wait_time:
        try:
            description = idx.describe_namespace(namespace=namespace)
            # Reset not_found_count on successful call
            not_found_count = 0
            # Handle both int and string types for record_count
            record_count = (
                int(description.record_count) if description.record_count is not None else 0
            )
            if record_count >= expected_count:
                logger.debug(
                    f"Namespace '{namespace}' has {record_count} records (expected {expected_count})"
                )
                return
            logger.debug(
                f"Polling namespace '{namespace}'. Current record_count: {record_count}, "
                f"expected: {expected_count}, waited: {time_waited}s"
            )
        except NotFoundException:
            # describe_namespace might be slightly behind, so allow a few retries
            not_found_count += 1
            if not_found_count >= max_not_found_retries:
                # If we've gotten NotFoundException multiple times, the namespace probably doesn't exist
                logger.debug(
                    f"Namespace '{namespace}' not found after {not_found_count} attempts. "
                    f"This may be expected in some tests."
                )
                raise
            logger.debug(
                f"Namespace '{namespace}' not found (attempt {not_found_count}/{max_not_found_retries}). "
                f"Retrying - describe_namespace might be slightly behind."
            )
        except Exception as e:
            logger.debug(f"Error describing namespace '{namespace}': {e}")

        time.sleep(wait_per_iteration)
        time_waited += wait_per_iteration

    # Check one more time before raising timeout
    try:
        description = idx.describe_namespace(namespace=namespace)
        record_count = int(description.record_count) if description.record_count is not None else 0
        if record_count >= expected_count:
            logger.debug(
                f"Namespace '{namespace}' has {record_count} records (expected {expected_count}) after timeout check"
            )
            return
        raise TimeoutError(
            f"Timeout waiting for namespace '{namespace}' to have {expected_count} records "
            f"after {time_waited} seconds. Current record_count: {record_count}"
        )
    except NotFoundException:
        # Re-raise NotFoundException as-is (expected in some tests)
        raise
    except Exception as e:
        if isinstance(e, TimeoutError):
            raise
        raise TimeoutError(
            f"Timeout waiting for namespace '{namespace}' to have {expected_count} records "
            f"after {time_waited} seconds. Error getting final count: {e}"
        )


@pytest.fixture(scope="session")
def seed_for_list(idx, list_namespace, wait=True):
    logger.debug(f"Upserting into list namespace '{list_namespace}'")
    response_infos = []
    for i in range(0, 1000, 50):
        response = idx.upsert(
            vectors=[(str(i + d), embedding_values(2)) for d in range(50)], namespace=list_namespace
        )
        response_infos.append(response._response_info)

    if wait:
        # Wait for the last batch's LSN to be reconciled
        poll_until_lsn_reconciled(idx, response_infos[-1], namespace=list_namespace)
        # Also wait for namespace to have the expected total count
        # This ensures all vectors are indexed, not just the last batch
        # Use try/except to handle cases where namespace might not exist yet
        try:
            poll_namespace_until_ready(idx, list_namespace, expected_count=1000, max_wait_time=120)
        except Exception as e:
            # If namespace doesn't exist or other error, log but don't fail
            # This can happen in tests that don't use the seeded namespace
            logger.debug(f"Could not poll namespace '{list_namespace}': {e}")

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
        # Poll to ensure vectors are available for listing
        poll_until_list_has_results(idx, prefix="99", namespace=list_namespace, expected_count=11)

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
        # Poll to ensure vectors are available for listing
        poll_until_list_has_results(idx, prefix="99", namespace=list_namespace, expected_count=11)

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
        # Poll to ensure vectors are available for listing
        poll_until_list_has_results(idx, prefix="99", namespace=list_namespace, expected_count=11)

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
        # Poll to ensure vectors are available for listing
        poll_until_list_has_results(idx, prefix="99", namespace=list_namespace, expected_count=11)

        vectors = []

        for ids in idx.list(prefix="99", limit=5, namespace=list_namespace):
            result = idx.fetch(ids=ids, namespace=list_namespace)
            vectors.extend([v for _, v in result.vectors.items()])

        assert len(vectors) == 11
        assert set([v.id for v in vectors]) == set(
            ["99", "990", "991", "992", "993", "994", "995", "996", "997", "998", "999"]
        )
