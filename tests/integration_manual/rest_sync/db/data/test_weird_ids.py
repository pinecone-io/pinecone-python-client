import os
import pytest
from tests.integration.helpers import random_string
from .weird_ids_setup import weird_valid_ids, weird_invalid_ids, setup_weird_ids_data
import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def weird_ids_namespace():
    return random_string(10)


@pytest.fixture(scope="class")
def seed_weird_ids(idx, weird_ids_namespace):
    logger.info("Seeding data in weird ids namespace " + weird_ids_namespace)
    setup_weird_ids_data(idx, weird_ids_namespace, True)
    yield


@pytest.mark.usefixtures("seed_weird_ids")
@pytest.mark.skipif(
    os.getenv("SKIP_WEIRD") == "true", reason="We don't need to run all of these every time"
)
class TestHandlingOfWeirdIds:
    def test_fetch_weird_ids(self, idx, weird_ids_namespace):
        weird_ids = weird_valid_ids()
        batch_size = 100
        for i in range(0, len(weird_ids), batch_size):
            ids_to_fetch = weird_ids[i : i + batch_size]
            results = idx.fetch(ids=ids_to_fetch, namespace=weird_ids_namespace)
            assert results.usage["read_units"] > 0
            assert len(results.vectors) == len(ids_to_fetch)
            for id in ids_to_fetch:
                assert id in results.vectors
                assert results.vectors[id].id == id
                assert results.vectors[id].metadata is None
                assert results.vectors[id].values is not None
                assert len(results.vectors[id].values) == 2

    @pytest.mark.parametrize("id_to_query", weird_valid_ids())
    def test_query_weird_ids(self, idx, weird_ids_namespace, id_to_query):
        results = idx.query(
            id=id_to_query, top_k=10, namespace=weird_ids_namespace, include_values=True
        )
        assert results.usage["read_units"] > 0
        assert len(results.matches) == 10
        assert results.namespace == weird_ids_namespace
        assert results.matches[0].id is not None
        assert results.matches[0].metadata is None
        assert results.matches[0].values is not None
        assert len(results.matches[0].values) == 2

    def test_list_weird_ids(self, idx, weird_ids_namespace):
        expected_ids = set(weird_valid_ids())
        id_iterator = idx.list(namespace=weird_ids_namespace)
        for page in id_iterator:
            for id in page:
                assert id in expected_ids

    @pytest.mark.parametrize("id_to_upsert", weird_invalid_ids())
    def test_weird_invalid_ids(self, idx, weird_ids_namespace, id_to_upsert):
        with pytest.raises(Exception) as e:
            idx.upsert(vectors=[(id_to_upsert, [0.1, 0.1])], namespace=weird_ids_namespace)
        assert "Vector ID must be ASCII" in str(e.value)

    def test_null_character(self, idx, weird_ids_namespace):
        with pytest.raises(Exception) as e:
            idx.upsert(vectors=[("\0", [0.1, 0.1])], namespace=weird_ids_namespace)

        assert "Vector ID must not contain null character" in str(e.value)
