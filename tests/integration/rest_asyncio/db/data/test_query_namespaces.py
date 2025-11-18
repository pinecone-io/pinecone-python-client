import pytest
from tests.integration.helpers import random_string
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async

from pinecone import Vector


@pytest.mark.asyncio
class TestQueryNamespacesRest:
    async def test_query_namespaces(self, index_host, metric):
        asyncio_idx = build_asyncioindex_client(index_host)

        ns_prefix = random_string(5)
        ns1 = f"{ns_prefix}-ns1"
        ns2 = f"{ns_prefix}-ns2"
        ns3 = f"{ns_prefix}-ns3"

        upsert1 = await asyncio_idx.upsert(
            vectors=[
                Vector(id="id1", values=[0.1, 0.2], metadata={"genre": "drama", "key": 1}),
                Vector(id="id2", values=[0.2, 0.3], metadata={"genre": "drama", "key": 2}),
                Vector(id="id3", values=[0.4, 0.5], metadata={"genre": "action", "key": 3}),
                Vector(id="id4", values=[0.6, 0.7], metadata={"genre": "action", "key": 4}),
            ],
            namespace=ns1,
        )
        upsert2 = await asyncio_idx.upsert(
            vectors=[
                Vector(id="id5", values=[0.21, 0.22], metadata={"genre": "drama", "key": 1}),
                Vector(id="id6", values=[0.22, 0.23], metadata={"genre": "drama", "key": 2}),
                Vector(id="id7", values=[0.24, 0.25], metadata={"genre": "action", "key": 3}),
                Vector(id="id8", values=[0.26, 0.27], metadata={"genre": "action", "key": 4}),
            ],
            namespace=ns2,
        )
        upsert3 = await asyncio_idx.upsert(
            vectors=[
                Vector(id="id9", values=[0.31, 0.32], metadata={"genre": "drama", "key": 1}),
                Vector(id="id10", values=[0.32, 0.33], metadata={"genre": "drama", "key": 2}),
                Vector(id="id11", values=[0.34, 0.35], metadata={"genre": "action", "key": 3}),
                Vector(id="id12", values=[0.36, 0.37], metadata={"genre": "action", "key": 4}),
            ],
            namespace=ns3,
        )

        await poll_until_lsn_reconciled_async(asyncio_idx, upsert1._response_info, namespace=ns1)
        await poll_until_lsn_reconciled_async(asyncio_idx, upsert2._response_info, namespace=ns2)
        await poll_until_lsn_reconciled_async(asyncio_idx, upsert3._response_info, namespace=ns3)

        results = await asyncio_idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[ns1, ns2, ns3],
            metric=metric,
            include_values=True,
            include_metadata=True,
            filter={"genre": {"$eq": "drama"}},
            top_k=100,
        )
        assert len(results.matches) == 6
        assert results.usage.read_units > 0
        for item in results.matches:
            assert item.metadata["genre"] == "drama"
        assert results.matches[0].id == "id1"
        assert results.matches[0].namespace == ns1

        # Using dot-style accessors
        assert results.matches[0].metadata["genre"] == "drama"
        assert results.matches[0].metadata["key"] == 1

        # Using dictionary-style accessors
        assert results.matches[0]["metadata"]["genre"] == "drama"
        assert results.matches[0]["metadata"]["key"] == 1

        # Using .get() accessors
        assert results.get("matches", [])[0].get("metadata", {}).get("genre") == "drama"
        assert results.matches[0].get("metadata", {}) == {"genre": "drama", "key": 1}
        assert results.matches[0].get("metadata", {}).get("genre") == "drama"

        assert results.matches[1].id == "id2"
        assert results.matches[1].namespace == ns1
        assert results.matches[2].id == "id5"
        assert results.matches[2].namespace == ns2

        # Non-existent namespace shouldn't cause any problem
        results2 = await asyncio_idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[ns1, ns2, ns3, f"{ns_prefix}-nonexistent"],
            metric=metric,
            include_values=True,
            include_metadata=True,
            filter={"genre": {"$eq": "action"}},
            top_k=100,
        )
        assert len(results2.matches) == 6
        assert results2.usage.read_units > 0
        for item in results2.matches:
            assert item.metadata["genre"] == "action"

        # Test with empty filter, top_k greater than number of results
        results3 = await asyncio_idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[ns1, ns2, ns3],
            metric=metric,
            include_values=True,
            include_metadata=True,
            filter={},
            top_k=100,
        )
        assert len(results3.matches) == 12
        assert results3.usage.read_units > 0

        # Test when all results are filtered out
        results4 = await asyncio_idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[ns1, ns2, ns3],
            metric=metric,
            include_values=True,
            include_metadata=True,
            filter={"genre": {"$eq": "comedy"}},
            top_k=100,
        )
        assert len(results4.matches) == 0
        assert results4.usage.read_units > 0

        # Test with top_k less than number of results
        results5 = await asyncio_idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[ns1, ns2, ns3],
            metric=metric,
            include_values=True,
            include_metadata=True,
            filter={},
            top_k=2,
        )
        assert len(results5.matches) == 2

        # Test when all namespaces are non-existent (same as all results filtered / empty)
        results6 = await asyncio_idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=[
                f"{ns_prefix}-nonexistent1",
                f"{ns_prefix}-nonexistent2",
                f"{ns_prefix}-nonexistent3",
            ],
            metric=metric,
            include_values=True,
            include_metadata=True,
            filter={"genre": {"$eq": "comedy"}},
            top_k=2,
        )
        assert len(results6.matches) == 0
        assert results6.usage.read_units > 0
        await asyncio_idx.close()

    async def test_single_result_per_namespace(self, index_host):
        asyncio_idx = build_asyncioindex_client(index_host)

        ns_prefix = random_string(5)
        ns1 = f"{ns_prefix}-ns1"
        ns2 = f"{ns_prefix}-ns2"

        upsert1 = await asyncio_idx.upsert(
            vectors=[
                Vector(id="id1", values=[0.1, 0.2], metadata={"genre": "drama", "key": 1}),
                Vector(id="id2", values=[0.2, 0.3], metadata={"genre": "drama", "key": 2}),
            ],
            namespace=ns1,
        )
        upsert2 = await asyncio_idx.upsert(
            vectors=[
                Vector(id="id5", values=[0.21, 0.22], metadata={"genre": "drama", "key": 1}),
                Vector(id="id6", values=[0.22, 0.23], metadata={"genre": "drama", "key": 2}),
            ],
            namespace=ns2,
        )

        await poll_until_lsn_reconciled_async(asyncio_idx, upsert1._response_info, namespace=ns1)
        await poll_until_lsn_reconciled_async(asyncio_idx, upsert2._response_info, namespace=ns2)

        results = await asyncio_idx.query_namespaces(
            vector=[0.1, 0.21],
            namespaces=[ns1, ns2],
            metric="cosine",
            include_values=True,
            include_metadata=True,
            filter={"key": {"$eq": 1}},
            top_k=2,
        )
        assert len(results.matches) == 2
        assert results.matches[0].id == "id1"
        assert results.matches[0].namespace == ns1
        assert results.matches[1].id == "id5"
        assert results.matches[1].namespace == ns2
        await asyncio_idx.close()

    async def test_missing_namespaces(self, index_host):
        asyncio_idx = build_asyncioindex_client(index_host)

        with pytest.raises(ValueError) as e:
            await asyncio_idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=[],
                metric="cosine",
                include_values=True,
                include_metadata=True,
                filter={},
                top_k=2,
            )
        assert str(e.value) == "At least one namespace must be specified"

        with pytest.raises(ValueError) as e:
            await asyncio_idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=None,
                metric="cosine",
                include_values=True,
                include_metadata=True,
                filter={},
                top_k=2,
            )
        assert str(e.value) == "At least one namespace must be specified"
        await asyncio_idx.close()

    async def test_missing_metric(self, index_host):
        asyncio_idx = build_asyncioindex_client(index_host)

        with pytest.raises(TypeError) as e:
            await asyncio_idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1"],
                include_values=True,
                include_metadata=True,
                filter={},
                top_k=2,
            )
        assert "query_namespaces() missing 1 required positional argument: 'metric'" in str(e.value)
        await asyncio_idx.close()
