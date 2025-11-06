import pytest
from pinecone import QueryResponse, Vector, FilterBuilder
from ..helpers import embedding_values, poll_until_lsn_reconciled, generate_name
import logging
from pinecone.grpc import GRPCIndex
from concurrent.futures import wait, ALL_COMPLETED


logger = logging.getLogger(__name__)


def find_by_id(matches, id):
    with_id = [match for match in matches if match.id == id]
    return with_id[0] if len(with_id) > 0 else None


@pytest.fixture(scope="session")
def query_namespace():
    return generate_name("query_namespace", "test")


def seed(idx, namespace):
    # Upsert without metadata
    logger.info(f"Seeding vectors without metadata into namespace '{namespace}'")
    upsert1 = idx.upsert(
        vectors=[
            ("1", embedding_values(2), {"test_file": "test_query_future.py"}),
            ("2", embedding_values(2), {"test_file": "test_query_future.py"}),
            ("3", embedding_values(2), {"test_file": "test_query_future.py"}),
        ],
        namespace=namespace,
        async_req=True,
    )

    # Upsert with metadata
    logger.info(f"Seeding vectors with metadata into namespace '{namespace}'")
    upsert2 = idx.upsert(
        vectors=[
            Vector(
                id="4",
                values=embedding_values(2),
                metadata={"genre": "action", "runtime": 120, "test_file": "test_query_future.py"},
            ),
            Vector(
                id="5",
                values=embedding_values(2),
                metadata={"genre": "comedy", "runtime": 90, "test_file": "test_query_future.py"},
            ),
            Vector(
                id="6",
                values=embedding_values(2),
                metadata={"genre": "romance", "runtime": 240, "test_file": "test_query_future.py"},
            ),
        ],
        namespace=namespace,
        async_req=True,
    )

    # Upsert with dict
    upsert3 = idx.upsert(
        vectors=[
            {
                "id": "7",
                "values": embedding_values(2),
                "metadata": {"test_file": "test_query_future.py"},
            },
            {
                "id": "8",
                "values": embedding_values(2),
                "metadata": {"test_file": "test_query_future.py"},
            },
            {
                "id": "9",
                "values": embedding_values(2),
                "metadata": {"test_file": "test_query_future.py"},
            },
        ],
        namespace=namespace,
        async_req=True,
    )

    wait([upsert1, upsert2, upsert3], timeout=10, return_when=ALL_COMPLETED)

    upsert_results = [upsert1.result(), upsert2.result(), upsert3.result()]
    for upsert_result in upsert_results:
        lsn_committed = upsert_result._response_info.get("lsn_committed")
        if lsn_committed is not None:
            poll_until_lsn_reconciled(idx, lsn_committed, namespace=namespace)


@pytest.fixture(scope="class")
def seed_for_query(idx, query_namespace):
    seed(idx, query_namespace)
    seed(idx, "")
    yield


@pytest.mark.usefixtures("seed_for_query")
@pytest.mark.parametrize("use_nondefault_namespace", [True, False])
class TestQueryAsync:
    def setup_method(self):
        self.expected_dimension = 2

    def test_query_by_id(
        self, idx: GRPCIndex, query_namespace: str, use_nondefault_namespace: bool
    ):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        query_future = idx.query(
            id="1",
            namespace=target_namespace,
            filter=FilterBuilder().eq("test_file", "test_query_future.py").build(),
            top_k=10,
            async_req=True,
        )

        done, not_done = wait([query_future], timeout=10, return_when=ALL_COMPLETED)

        assert len(done) == 1
        assert len(not_done) == 0

        query_result = query_future.result()

        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace

        assert query_result.usage is not None
        assert query_result.usage["read_units"] is not None
        assert query_result.usage["read_units"] > 0

        # By default, does not include values or metadata
        record_with_metadata = find_by_id(query_result.matches, "4")
        assert record_with_metadata.metadata is None
        assert record_with_metadata.values == []

    def test_query_by_vector(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        query_result = idx.query(
            vector=embedding_values(2), namespace=target_namespace, top_k=10, async_req=True
        ).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace

    def test_query_by_vector_include_values(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        query_result = idx.query(
            vector=embedding_values(2),
            namespace=target_namespace,
            include_values=True,
            top_k=10,
            async_req=True,
        ).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace
        assert len(query_result.matches) > 0
        assert query_result.matches[0].values is not None
        assert len(query_result.matches[0].values) == self.expected_dimension

    def test_query_by_vector_include_metadata(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        query_result = idx.query(
            vector=embedding_values(2),
            namespace=target_namespace,
            include_metadata=True,
            top_k=10,
            filter=FilterBuilder().eq("test_file", "test_query_future.py").build(),
            async_req=True,
        ).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace

        matches_with_metadata = [
            match
            for match in query_result.matches
            if match.metadata is not None and match.metadata != {}
        ]
        # Check that we have at least the vectors we seeded
        assert len(matches_with_metadata) >= 3
        assert find_by_id(query_result.matches, "4") is not None
        assert find_by_id(query_result.matches, "4").metadata["genre"] == "action"

    def test_query_by_vector_include_values_and_metadata(
        self, idx, query_namespace, use_nondefault_namespace
    ):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        query_result = idx.query(
            vector=embedding_values(2),
            namespace=target_namespace,
            filter=FilterBuilder().eq("test_file", "test_query_future.py").build(),
            include_values=True,
            include_metadata=True,
            top_k=10,
            async_req=True,
        ).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace

        matches_with_metadata = [
            match
            for match in query_result.matches
            if match.metadata is not None and match.metadata != {}
        ]
        # Check that we have at least the vectors we seeded
        assert len(matches_with_metadata) >= 3
        assert find_by_id(query_result.matches, "4") is not None
        assert find_by_id(query_result.matches, "4").metadata["genre"] == "action"
        assert len(query_result.matches[0].values) == self.expected_dimension


class TestQueryEdgeCasesAsync:
    def test_query_in_empty_namespace(self, idx):
        query_result = idx.query(id="1", namespace="empty", top_k=10, async_req=True).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == "empty"
        assert len(query_result.matches) == 0


@pytest.mark.usefixtures("seed_for_query")
@pytest.mark.parametrize("use_nondefault_namespace", [True, False])
class TestQueryWithFilterAsync:
    def test_query_by_id_with_filter(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        query_result = idx.query(
            id="1", namespace=target_namespace, filter={"genre": "action"}, top_k=10, async_req=True
        ).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace
        # Check that we have at least the vector we seeded
        assert len(query_result.matches) >= 1
        assert find_by_id(query_result.matches, "4") is not None

    def test_query_by_id_with_filter_gt(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        query_result = idx.query(
            id="1",
            namespace=target_namespace,
            filter={"runtime": {"$gt": 100}},
            top_k=10,
            async_req=True,
        ).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace
        assert len(query_result.matches) == 2
        assert find_by_id(query_result.matches, "4") is not None
        assert find_by_id(query_result.matches, "6") is not None

    def test_query_by_id_with_filter_gte(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        query_result = idx.query(
            id="1",
            namespace=target_namespace,
            filter={"runtime": {"$gte": 90}},
            top_k=10,
            async_req=True,
        ).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace
        assert len(query_result.matches) == 3
        assert find_by_id(query_result.matches, "4") is not None
        assert find_by_id(query_result.matches, "5") is not None
        assert find_by_id(query_result.matches, "6") is not None

    def test_query_by_id_with_filter_lt(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        query_result = idx.query(
            id="1",
            namespace=target_namespace,
            filter={"runtime": {"$lt": 100}},
            top_k=10,
            async_req=True,
        ).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace
        assert len(query_result.matches) == 1
        assert find_by_id(query_result.matches, "5") is not None

    def test_query_by_id_with_filter_lte(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        query_result = idx.query(
            id="1",
            namespace=target_namespace,
            filter={"runtime": {"$lte": 120}},
            top_k=10,
            async_req=True,
        ).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace
        assert len(query_result.matches) == 2
        assert find_by_id(query_result.matches, "4") is not None
        assert find_by_id(query_result.matches, "5") is not None

    def test_query_by_id_with_filter_in(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        query_result = idx.query(
            id="1",
            namespace=target_namespace,
            filter={"genre": {"$in": ["romance"]}},
            top_k=10,
            async_req=True,
        ).result()
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace
        assert len(query_result.matches) == 1
        assert find_by_id(query_result.matches, "6") is not None

    def test_query_by_id_with_filter_nin(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        query_result = idx.query(
            id="1",
            namespace=target_namespace,
            filter=FilterBuilder().nin("genre", ["romance"]).build(),
            include_metadata=True,
            top_k=10,
            async_req=True,
        ).result()

        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace

        matches_with_metadata = [
            match
            for match in query_result.matches
            if match.metadata is not None and match.metadata.get("genre") is not None
        ]
        # Check that we have at least the vectors we seeded
        assert len(matches_with_metadata) >= 2
        for match in matches_with_metadata:
            assert match.metadata["genre"] != "romance"

    def test_query_by_id_with_filter_eq(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        query_result = idx.query(
            id="1",
            namespace=target_namespace,
            filter={"genre": {"$eq": "action"}},
            include_metadata=True,
            top_k=10,
            async_req=True,
        ).result()

        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace
        for match in query_result.matches:
            logger.info(f"Match: id: {match.id} metadata: {match.metadata}")

        matches_with_metadata = [
            match
            for match in query_result.matches
            if match.metadata is not None and match.metadata.get("genre") is not None
        ]
        # Check that we have at least the vector we seeded
        assert len(matches_with_metadata) >= 1
        # Verify that vector "4" is in the results
        assert find_by_id(query_result.matches, "4") is not None
        assert find_by_id(query_result.matches, "4").metadata["genre"] == "action"

    def test_query_by_id_with_filter_ne(self, idx, query_namespace, use_nondefault_namespace):
        target_namespace = query_namespace if use_nondefault_namespace else ""

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        query_result = idx.query(
            id="1",
            namespace=target_namespace,
            filter={"genre": {"$ne": "action"}},
            include_metadata=True,
            top_k=10,
            async_req=True,
        ).result()
        for match in query_result.matches:
            logger.info(f"Match: id: {match.id} metadata: {match.metadata}")
        assert isinstance(query_result, QueryResponse) == True
        assert query_result.namespace == target_namespace

        matches_with_metadata = [
            match
            for match in query_result.matches
            if match.metadata is not None and match.metadata.get("genre") is not None
        ]
        # Check that we have at least the vectors we seeded
        assert len(matches_with_metadata) >= 2
        # Verify that vectors "5" and "6" are in the results
        assert find_by_id(query_result.matches, "5") is not None
        assert find_by_id(query_result.matches, "6") is not None
        for match in matches_with_metadata:
            assert match.metadata["genre"] != "action"
            assert match.id != "4"
