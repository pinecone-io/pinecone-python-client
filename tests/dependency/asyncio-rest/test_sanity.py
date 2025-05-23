import pytest
import os
import asyncio
from pinecone import PineconeAsyncio

import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def index_name():
    name = os.environ.get("INDEX_NAME", None)
    if name is None or name == "":
        raise "INDEX_NAME environment variable is not set"
    return name


@pytest.mark.asyncio
class TestSanityAsyncioRest:
    async def test_sanity(self, index_name):
        async with PineconeAsyncio() as pc:
            logger.info("Testing with index name: " + index_name)
            assert index_name != ""

            # Verify index exists with expected properties
            available_indexes = await pc.list_indexes()
            assert index_name in available_indexes.names()

            description = await pc.describe_index(name=index_name)
            assert description.dimension == 2
            logger.info("Index description: %s", description)

            async with pc.IndexAsyncio(host=description.host) as idx:
                resp = await idx.upsert(
                    vectors=[("1", [1.0, 2.0]), ("2", [3.0, 4.0]), ("3", [5.0, 6.0])]
                )
                logger.info("Upsert response: %s", resp)

                # Wait for index freshness
                await asyncio.sleep(30)

                # Check the vector count reflects some data has been upserted
                description = await idx.describe_index_stats()
                logger.info("Index stats: %s", description)
                assert description.dimension == 2
                assert description.total_vector_count >= 3

                # Query for results
                query_results = await idx.query(id="1", top_k=10, include_values=True)
                logger.info("Query results: %s", query_results)
                assert query_results.matches[0].id == "1"
                assert len(query_results.matches) >= 3

                # Call a bulk import api method, should not raise an exception
                async for i in idx.list_imports():
                    assert i is not None

                # Call an inference method, should not raise an exception
                from pinecone import EmbedModel

                resp = await pc.inference.embed(
                    model=EmbedModel.Multilingual_E5_Large,
                    inputs=["Hello, how are you?", "I am doing well, thank you for asking."],
                    parameters={"input_type": "passage", "truncate": "END"},
                )
                logger.info("Embed response: %s", resp)
