import pytest
from pinecone import PineconeException
from ..helpers import embedding_values, poll_until_lsn_reconciled


@pytest.fixture(scope="session")
def query_error_namespace():
    return "query-error-namespace"


@pytest.fixture(scope="session")
def seed_for_query_error_cases(idx, query_error_namespace):
    upsert1 = idx.upsert(
        vectors=[
            ("1", embedding_values(2)),
            ("2", embedding_values(2)),
            ("3", embedding_values(2)),
        ],
        namespace=query_error_namespace,
    )
    upsert2 = idx.upsert(
        vectors=[
            ("4", embedding_values(2)),
            ("5", embedding_values(2)),
            ("6", embedding_values(2)),
        ],
        namespace="__default__",
    )
    poll_until_lsn_reconciled(idx, upsert1._response_info, namespace=query_error_namespace)
    poll_until_lsn_reconciled(idx, upsert2._response_info, namespace="__default__")
    yield


@pytest.mark.usefixtures("seed_for_query_error_cases")
@pytest.mark.parametrize("use_nondefault_namespace", [True, False])
class TestQueryErrorCases:
    def test_query_with_invalid_vector(self, idx, query_error_namespace, use_nondefault_namespace):
        target_namespace = query_error_namespace if use_nondefault_namespace else ""

        with pytest.raises(PineconeException) as e:
            idx.query(vector=[0.23, 2.23, 3.43], namespace=target_namespace, top_k=10)

        assert "vector" in str(e.value).lower()

    def test_query_with_invalid_id(self, idx, query_error_namespace, use_nondefault_namespace):
        target_namespace = query_error_namespace if use_nondefault_namespace else ""

        with pytest.raises(TypeError):
            idx.query(id=1, namespace=target_namespace, top_k=10)

    def test_query_with_invalid_top_k(self, idx, query_error_namespace, use_nondefault_namespace):
        target_namespace = query_error_namespace if use_nondefault_namespace else ""

        with pytest.raises((PineconeException, ValueError)):
            idx.query(id="1", namespace=target_namespace, top_k=-1)

    def test_query_with_missing_top_k(self, idx, query_error_namespace, use_nondefault_namespace):
        target_namespace = query_error_namespace if use_nondefault_namespace else ""

        with pytest.raises((TypeError, PineconeException)) as e:
            idx.query(id="1", namespace=target_namespace)

        assert "top_k" in str(e.value).lower()
