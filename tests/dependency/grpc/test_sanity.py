import pytest
import os
import time
from pinecone.grpc import PineconeGRPC
import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def index_name():
    name = os.environ.get("INDEX_NAME", None)
    if name is None:
        raise "INDEX_NAME environment variable is not set"
    return name


@pytest.fixture
def client():
    return PineconeGRPC(api_key=os.environ.get("PINECONE_API_KEY"))


class TestSanityRest:
    def test_sanity(self, index_name, client):
        logger.info("Testing with index name: %s", index_name)
        assert index_name != ""

        # Verify index exists with expected properties
        assert index_name in client.list_indexes().names()
        description = client.describe_index(name=index_name)
        assert description.dimension == 2
        logger.info("Index description: %s", description)

        idx = client.Index(index_name)
        resp = idx.upsert(vectors=[("1", [1.0, 2.0]), ("2", [3.0, 4.0]), ("3", [5.0, 6.0])])
        logger.info("Upsert response: %s", resp)

        # Wait for index freshness
        time.sleep(30)

        # Check the vector count reflects some data has been upserted
        description = idx.describe_index_stats()
        logger.info("Index stats: %s", description)
        assert description.dimension == 2
        assert description.total_vector_count >= 3

        # Query for results
        query_results = idx.query(id="1", top_k=10, include_values=True)
        logger.info("Query results: %s", query_results)
        assert query_results.matches[0].id == "1"
        assert len(query_results.matches) == 3
