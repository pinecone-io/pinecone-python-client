import logging
import pytest
import pytest_asyncio
import asyncio
from ..helpers import embedding_values, random_string
from pinecone import Vector, FetchByMetadataResponse

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fetch_by_metadata_namespace():
    return random_string(10)


async def seed_for_fetch_by_metadata(idx, namespace):
    """Seed vectors with various metadata for testing fetch_by_metadata."""
    logger.info(f"Seeding vectors with metadata into namespace '{namespace}'")

    # Upsert vectors with different metadata
    await idx.upsert(
        vectors=[
            Vector(
                id="genre-action-1",
                values=embedding_values(2),
                metadata={"genre": "action", "year": 2020, "rating": 8.5},
            ),
            Vector(
                id="genre-action-2",
                values=embedding_values(2),
                metadata={"genre": "action", "year": 2021, "rating": 7.5},
            ),
            Vector(
                id="genre-comedy-1",
                values=embedding_values(2),
                metadata={"genre": "comedy", "year": 2020, "rating": 9.0},
            ),
            Vector(
                id="genre-comedy-2",
                values=embedding_values(2),
                metadata={"genre": "comedy", "year": 2022, "rating": 8.0},
            ),
            Vector(
                id="genre-drama-1",
                values=embedding_values(2),
                metadata={"genre": "drama", "year": 2020, "rating": 9.5},
            ),
            Vector(
                id="genre-romance-1",
                values=embedding_values(2),
                metadata={"genre": "romance", "year": 2021, "rating": 7.0},
            ),
            Vector(id="no-metadata-1", values=embedding_values(2), metadata=None),
        ],
        namespace=namespace,
    )

    # Wait for vectors to be available by polling fetch_by_metadata
    max_wait = 60
    wait_time = 0
    while wait_time < max_wait:
        try:
            results = await idx.fetch_by_metadata(
                filter={"genre": {"$in": ["action", "comedy", "drama", "romance"]}},
                namespace=namespace,
                limit=10,
            )
            if len(results.vectors) >= 6:  # At least 6 vectors with genre metadata
                break
        except Exception:
            pass
        await asyncio.sleep(2)
        wait_time += 2


@pytest_asyncio.fixture(scope="function")
async def seed_for_fetch_by_metadata_fixture(idx, fetch_by_metadata_namespace):
    await seed_for_fetch_by_metadata(idx, fetch_by_metadata_namespace)
    await seed_for_fetch_by_metadata(idx, "")
    yield


@pytest.mark.usefixtures("seed_for_fetch_by_metadata_fixture")
class TestFetchByMetadataAsyncio:
    def setup_method(self):
        self.expected_dimension = 2

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    @pytest.mark.asyncio
    async def test_fetch_by_metadata_simple_filter(
        self, idx, fetch_by_metadata_namespace, use_nondefault_namespace
    ):
        target_namespace = fetch_by_metadata_namespace if use_nondefault_namespace else ""

        results = await idx.fetch_by_metadata(
            filter={"genre": {"$eq": "action"}}, namespace=target_namespace
        )
        assert isinstance(results, FetchByMetadataResponse)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 2
        assert "genre-action-1" in results.vectors
        assert "genre-action-2" in results.vectors

        # Verify metadata
        assert results.vectors["genre-action-1"].metadata["genre"] == "action"
        assert results.vectors["genre-action-2"].metadata["genre"] == "action"

        # Verify usage
        assert results.usage is not None
        assert results.usage["read_units"] is not None
        assert results.usage["read_units"] > 0

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    @pytest.mark.asyncio
    async def test_fetch_by_metadata_with_limit(
        self, idx, fetch_by_metadata_namespace, use_nondefault_namespace
    ):
        target_namespace = fetch_by_metadata_namespace if use_nondefault_namespace else ""

        results = await idx.fetch_by_metadata(
            filter={"genre": {"$eq": "action"}}, namespace=target_namespace, limit=1
        )
        assert isinstance(results, FetchByMetadataResponse)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 1

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    @pytest.mark.asyncio
    async def test_fetch_by_metadata_with_in_operator(
        self, idx, fetch_by_metadata_namespace, use_nondefault_namespace
    ):
        target_namespace = fetch_by_metadata_namespace if use_nondefault_namespace else ""

        results = await idx.fetch_by_metadata(
            filter={"genre": {"$in": ["comedy", "drama"]}}, namespace=target_namespace
        )
        assert isinstance(results, FetchByMetadataResponse)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 3  # comedy-1, comedy-2, drama-1
        assert "genre-comedy-1" in results.vectors
        assert "genre-comedy-2" in results.vectors
        assert "genre-drama-1" in results.vectors

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    @pytest.mark.asyncio
    async def test_fetch_by_metadata_with_multiple_conditions(
        self, idx, fetch_by_metadata_namespace, use_nondefault_namespace
    ):
        target_namespace = fetch_by_metadata_namespace if use_nondefault_namespace else ""

        results = await idx.fetch_by_metadata(
            filter={"genre": {"$eq": "action"}, "year": {"$eq": 2020}}, namespace=target_namespace
        )
        assert isinstance(results, FetchByMetadataResponse)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 1
        assert "genre-action-1" in results.vectors
        assert results.vectors["genre-action-1"].metadata["year"] == 2020

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    @pytest.mark.asyncio
    async def test_fetch_by_metadata_with_numeric_filter(
        self, idx, fetch_by_metadata_namespace, use_nondefault_namespace
    ):
        target_namespace = fetch_by_metadata_namespace if use_nondefault_namespace else ""

        results = await idx.fetch_by_metadata(
            filter={"year": {"$gte": 2021}}, namespace=target_namespace
        )
        assert isinstance(results, FetchByMetadataResponse)
        assert results.namespace == target_namespace
        # Should return action-2, comedy-2, romance-1 (all year >= 2021)
        assert len(results.vectors) >= 3
        assert "genre-action-2" in results.vectors
        assert "genre-comedy-2" in results.vectors
        assert "genre-romance-1" in results.vectors

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    @pytest.mark.asyncio
    async def test_fetch_by_metadata_no_results(
        self, idx, fetch_by_metadata_namespace, use_nondefault_namespace
    ):
        target_namespace = fetch_by_metadata_namespace if use_nondefault_namespace else ""

        results = await idx.fetch_by_metadata(
            filter={"genre": {"$eq": "horror"}}, namespace=target_namespace
        )
        assert isinstance(results, FetchByMetadataResponse)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 0

    @pytest.mark.asyncio
    async def test_fetch_by_metadata_nonexistent_namespace(self, idx):
        target_namespace = "nonexistent-namespace"

        results = await idx.fetch_by_metadata(
            filter={"genre": {"$eq": "action"}}, namespace=target_namespace
        )
        assert isinstance(results, FetchByMetadataResponse)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 0

    @pytest.mark.asyncio
    async def test_fetch_by_metadata_unspecified_namespace(self, idx):
        # Fetch without specifying namespace gives default namespace results
        results = await idx.fetch_by_metadata(filter={"genre": {"$eq": "action"}})
        assert isinstance(results, FetchByMetadataResponse)
        assert results.namespace == ""
        assert len(results.vectors) == 2

    @pytest.mark.asyncio
    async def test_fetch_by_metadata_pagination(self, idx, fetch_by_metadata_namespace):
        # First page
        results1 = await idx.fetch_by_metadata(
            filter={"genre": {"$in": ["action", "comedy", "drama", "romance"]}},
            namespace=fetch_by_metadata_namespace,
            limit=2,
        )
        assert isinstance(results1, FetchByMetadataResponse)
        assert len(results1.vectors) == 2

        # Check if pagination token exists (if more results available)
        if results1.pagination and results1.pagination.next:
            # Second page
            results2 = await idx.fetch_by_metadata(
                filter={"genre": {"$in": ["action", "comedy", "drama", "romance"]}},
                namespace=fetch_by_metadata_namespace,
                limit=2,
                pagination_token=results1.pagination.next,
            )
            assert isinstance(results2, FetchByMetadataResponse)
            assert len(results2.vectors) >= 0  # Could be 0 if no more results

            # Verify no overlap between pages
            page1_ids = set(results1.vectors.keys())
            page2_ids = set(results2.vectors.keys())
            assert len(page1_ids.intersection(page2_ids)) == 0
