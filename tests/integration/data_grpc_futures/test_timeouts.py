import pytest
from pinecone import QueryResponse, UpsertResponse, FetchResponse, Vector, PineconeException
from ..helpers import embedding_values
from .stub_backend import create_sleepy_test_server
import logging
from pinecone.grpc import GRPCIndex, PineconeGRPC
from concurrent.futures import wait, ALL_COMPLETED

logger = logging.getLogger(__name__)

SERVER_SLEEP_SECONDS = 1


@pytest.fixture(scope="session")
def grpc_server():
    logger.info("Starting gRPC test server")
    server = create_sleepy_test_server(port=50051, sleep_seconds=SERVER_SLEEP_SECONDS)
    yield server
    logger.info("Stopping gRPC test server")
    server.stop(0)


@pytest.fixture(scope="session")
def local_idx():
    pc = PineconeGRPC(api_key="test", ssl_verify=False)
    idx = pc.Index(host="localhost:50051")
    return idx


@pytest.mark.usefixtures("grpc_server")
class TestGrpcAsyncTimeouts_QueryByID:
    def test_query_by_id_with_custom_timeout_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS - 0.5
        query_results = local_idx.query(
            id="1", namespace="testnamespace", top_k=10, async_req=True, timeout=deadline
        )

        assert query_results._default_timeout == deadline
        done, not_done = wait(
            [query_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        with pytest.raises(PineconeException) as e:
            query_results.result()

        assert "Deadline Exceeded" in str(e.value)

    def test_query_by_id_with_custom_timeout_not_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS + 1
        query_results = local_idx.query(
            id="1", namespace="testnamespace", top_k=10, async_req=True, timeout=deadline
        )

        assert query_results._default_timeout == deadline
        done, not_done = wait(
            [query_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        result = query_results.result()
        assert result is not None
        assert result.usage.read_units == 1

    def test_query_by_id_with_default_timeout(self, local_idx: GRPCIndex):
        query_results = local_idx.query(id="1", namespace="testnamespace", top_k=10, async_req=True)

        # Default timeout is 5 seconds, which is longer than the test server sleep
        assert query_results._default_timeout == 5

        done, not_done = wait([query_results], timeout=10, return_when=ALL_COMPLETED)

        assert len(done) == 1
        assert len(not_done) == 0

        result = query_results.result()
        assert result is not None
        assert isinstance(result, QueryResponse)
        assert result.usage.read_units == 1
        assert result.matches[0].id == "1"


@pytest.mark.usefixtures("grpc_server")
class TestGrpcAsyncTimeouts_QueryByVector:
    def test_query_by_vector_with_timeout_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS - 0.5

        query_results = local_idx.query(
            vector=embedding_values(2),
            namespace="testnamespace",
            top_k=10,
            async_req=True,
            timeout=deadline,
        )

        assert query_results._default_timeout == deadline
        done, not_done = wait(
            [query_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        with pytest.raises(PineconeException) as e:
            query_results.result()

        assert "Deadline Exceeded" in str(e.value)

    def test_query_by_vector_with_custom_timeout_not_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS + 1
        query_results = local_idx.query(
            vector=embedding_values(2),
            namespace="testnamespace",
            top_k=10,
            async_req=True,
            timeout=deadline,
        )

        assert query_results._default_timeout == deadline
        done, not_done = wait(
            [query_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        result = query_results.result()
        assert result is not None
        assert isinstance(result, QueryResponse)
        assert result.usage.read_units == 1
        assert result.matches[0].id == "1"

    def test_query_by_vector_with_default_timeout(self, local_idx: GRPCIndex):
        query_results = local_idx.query(
            vector=embedding_values(2), namespace="testnamespace", top_k=10, async_req=True
        )

        # Default timeout is 5 seconds, which is longer than the test server sleep
        assert query_results._default_timeout == 5

        done, not_done = wait([query_results], timeout=10, return_when=ALL_COMPLETED)

        assert len(done) == 1
        assert len(not_done) == 0

        result = query_results.result()
        assert result is not None
        assert result.usage.read_units == 1


@pytest.mark.usefixtures("grpc_server")
class TestGrpcAsyncTimeouts_Upsert:
    def test_upsert_with_custom_timeout_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS - 0.5

        upsert_results = local_idx.upsert(
            vectors=[Vector(id="1", values=embedding_values(2), metadata={"genre": "action"})],
            namespace="testnamespace",
            async_req=True,
            timeout=deadline,
        )

        assert upsert_results._default_timeout == deadline
        done, not_done = wait(
            [upsert_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        with pytest.raises(PineconeException) as e:
            upsert_results.result()

        assert "Deadline Exceeded" in str(e.value)

    def test_upsert_with_custom_timeout_not_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS + 1
        upsert_results = local_idx.upsert(
            vectors=[Vector(id="1", values=embedding_values(2), metadata={"genre": "action"})],
            namespace="testnamespace",
            async_req=True,
            timeout=deadline,
        )

        assert upsert_results._default_timeout == deadline
        done, not_done = wait(
            [upsert_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        result = upsert_results.result()
        assert result is not None
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 1

    def test_upsert_with_default_timeout(self, local_idx: GRPCIndex):
        upsert_results = local_idx.upsert(
            vectors=[Vector(id="1", values=embedding_values(2), metadata={"genre": "action"})],
            namespace="testnamespace",
            async_req=True,
        )

        # Default timeout is 5 seconds, which is longer than the test server sleep
        assert upsert_results._default_timeout == 5

        done, not_done = wait([upsert_results], timeout=10, return_when=ALL_COMPLETED)

        assert len(done) == 1
        assert len(not_done) == 0

        result = upsert_results.result()
        assert result is not None
        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 1


@pytest.mark.usefixtures("grpc_server")
class TestGrpcAsyncTimeouts_Update:
    def test_update_with_custom_timeout_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS - 0.5

        update_results = local_idx.update(
            id="1",
            namespace="testnamespace",
            values=embedding_values(2),
            async_req=True,
            timeout=deadline,
        )

        assert update_results._default_timeout == deadline
        done, not_done = wait(
            [update_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        with pytest.raises(PineconeException) as e:
            update_results.result()

        assert "Deadline Exceeded" in str(e.value)

    def test_update_with_default_timeout(self, local_idx: GRPCIndex):
        update_results = local_idx.update(
            id="1", namespace="testnamespace", values=embedding_values(2), async_req=True
        )

        # Default timeout is 5 seconds, which is longer than the test server sleep
        assert update_results._default_timeout == 5

        done, not_done = wait([update_results], timeout=10, return_when=ALL_COMPLETED)

        assert len(done) == 1
        assert len(not_done) == 0

        result = update_results.result()
        assert result is not None
        assert result == {}

    def test_update_with_custom_timeout_not_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS + 1
        update_results = local_idx.update(
            id="1",
            namespace="testnamespace",
            values=embedding_values(2),
            async_req=True,
            timeout=deadline,
        )

        assert update_results._default_timeout == deadline
        done, not_done = wait(
            [update_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        result = update_results.result()
        assert result is not None
        assert result == {}


@pytest.mark.usefixtures("grpc_server")
class TestGrpcAsyncTimeouts_Delete:
    def test_delete_with_custom_timeout_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS - 0.5

        delete_results = local_idx.delete(
            ids=["1", "2", "3"], namespace="testnamespace", async_req=True, timeout=deadline
        )

        assert delete_results._default_timeout == deadline
        done, not_done = wait(
            [delete_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        with pytest.raises(PineconeException) as e:
            delete_results.result()

        assert "Deadline Exceeded" in str(e.value)

    def test_delete_with_custom_timeout_not_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS + 1
        delete_results = local_idx.delete(
            ids=["1", "2", "3"], namespace="testnamespace", async_req=True, timeout=deadline
        )

        assert delete_results._default_timeout == deadline
        done, not_done = wait(
            [delete_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        result = delete_results.result()
        assert result is not None
        assert result == {}

    def test_delete_with_default_timeout(self, local_idx: GRPCIndex):
        delete_results = local_idx.delete(
            ids=["1", "2", "3"], namespace="testnamespace", async_req=True
        )

        # Default timeout is 5 seconds, which is longer than the test server sleep
        assert delete_results._default_timeout == 5

        done, not_done = wait([delete_results], timeout=10, return_when=ALL_COMPLETED)

        assert len(done) == 1
        assert len(not_done) == 0

        result = delete_results.result()
        assert result is not None
        assert result == {}


@pytest.mark.usefixtures("grpc_server")
class TestGrpcAsyncTimeouts_Fetch:
    def test_fetch_with_custom_timeout_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS - 0.5

        fetch_results = local_idx.fetch(
            ids=["1", "2", "3"], namespace="testnamespace", async_req=True, timeout=deadline
        )

        assert fetch_results._default_timeout == deadline
        done, not_done = wait(
            [fetch_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        with pytest.raises(PineconeException) as e:
            fetch_results.result()

        assert "Deadline Exceeded" in str(e.value)

    def test_fetch_with_custom_timeout_not_exceeded(self, local_idx: GRPCIndex):
        deadline = SERVER_SLEEP_SECONDS + 1
        fetch_results = local_idx.fetch(
            ids=["1", "2", "3"], namespace="testnamespace", async_req=True, timeout=deadline
        )

        assert fetch_results._default_timeout == deadline
        done, not_done = wait(
            [fetch_results], timeout=SERVER_SLEEP_SECONDS + 1, return_when=ALL_COMPLETED
        )

        assert len(done) == 1
        assert len(not_done) == 0

        result = fetch_results.result()
        assert result is not None
        assert isinstance(result, FetchResponse)

    def test_fetch_with_default_timeout(self, local_idx: GRPCIndex):
        fetch_results = local_idx.fetch(
            ids=["1", "2", "3"], namespace="testnamespace", async_req=True
        )

        # Default timeout is 5 seconds, which is longer than the test server sleep
        assert fetch_results._default_timeout == 5

        done, not_done = wait([fetch_results], timeout=10, return_when=ALL_COMPLETED)

        assert len(done) == 1
        assert len(not_done) == 0

        result = fetch_results.result()
        assert result is not None
        assert isinstance(result, FetchResponse)

        assert result.vectors["1"].id == "1"
        assert result.vectors["2"].id == "2"
        assert result.vectors["3"].id == "3"
        assert result.usage.read_units == 1
        assert result.namespace == "testnamespace"
