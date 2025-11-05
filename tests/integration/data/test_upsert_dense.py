import pytest
from pinecone import Vector
from ..helpers import (
    poll_stats_for_namespace,
    poll_until_lsn_reconciled,
    embedding_values,
    random_string,
)


@pytest.fixture(scope="session")
def upsert_dense_namespace():
    return random_string(10)


class TestUpsertDense:
    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_upsert_to_namespace(self, idx, upsert_dense_namespace, use_nondefault_namespace):
        target_namespace = upsert_dense_namespace if use_nondefault_namespace else ""

        # Upsert with tuples
        response1 = idx.upsert(
            vectors=[
                ("1", embedding_values()),
                ("2", embedding_values()),
                ("3", embedding_values()),
            ],
            namespace=target_namespace,
        )
        committed_lsn = None
        if hasattr(response1, "_response_info") and response1._response_info:
            committed_lsn = response1._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response1._response_info is not None
            ), "Expected _response_info to be present on upsert response"

        # Upsert with objects
        response2 = idx.upsert(
            vectors=[
                Vector(id="4", values=embedding_values()),
                Vector(id="5", values=embedding_values()),
                Vector(id="6", values=embedding_values()),
            ],
            namespace=target_namespace,
        )
        if hasattr(response2, "_response_info") and response2._response_info:
            committed_lsn2 = response2._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response2._response_info is not None
            ), "Expected _response_info to be present on upsert response"
            if committed_lsn2 is not None:
                committed_lsn = committed_lsn2

        # Upsert with dict
        response3 = idx.upsert(
            vectors=[
                {"id": "7", "values": embedding_values()},
                {"id": "8", "values": embedding_values()},
                {"id": "9", "values": embedding_values()},
            ],
            namespace=target_namespace,
        )
        if hasattr(response3, "_response_info") and response3._response_info:
            committed_lsn3 = response3._response_info.get("lsn_committed")
            # Assert that _response_info is present when we extract LSN
            assert (
                response3._response_info is not None
            ), "Expected _response_info to be present on upsert response"
            if committed_lsn3 is not None:
                committed_lsn = committed_lsn3

        # Use LSN-based polling if available, otherwise fallback to stats polling
        if committed_lsn is not None:

            def check_namespace_count():
                stats = idx.describe_index_stats()
                if target_namespace == "":
                    namespace_key = "__default__" if "__default__" in stats.namespaces else ""
                else:
                    namespace_key = target_namespace
                return (
                    namespace_key in stats.namespaces
                    and stats.namespaces[namespace_key].vector_count >= 9
                )

            poll_until_lsn_reconciled(
                idx,
                committed_lsn,
                operation_name="upsert_to_namespace",
                check_fn=check_namespace_count,
            )
        else:
            poll_stats_for_namespace(idx, target_namespace, 9)

        # Check the vector count reflects some data has been upserted
        stats = idx.describe_index_stats()
        assert stats.total_vector_count >= 9
        # The default namespace may be represented as "" or "__default__" in the API response
        if target_namespace == "":
            namespace_key = "__default__" if "__default__" in stats.namespaces else ""
        else:
            namespace_key = target_namespace
        assert namespace_key in stats.namespaces
        assert stats.namespaces[namespace_key].vector_count == 9
