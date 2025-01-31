import pytest
from pinecone.exceptions import PineconeApiException
from pinecone import PineconeAsyncio


@pytest.mark.asyncio
class TestSparseIndex:
    async def test_create_sparse_index_with_metric(self, create_sl_index_params):
        pc = PineconeAsyncio()

        create_sl_index_params["metric"] = "dotproduct"

        create_sl_index_params["vector_type"] = "sparse"
        del create_sl_index_params["dimension"]

        await pc.create_index(**create_sl_index_params)
        desc = await pc.describe_index(create_sl_index_params["name"])
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"
        assert desc.dimension is None
        assert desc.deletion_protection == "disabled"  # default value

        for i in await pc.list_indexes():
            if i.name == create_sl_index_params["name"]:
                assert i.metric == "dotproduct"
                assert i.vector_type == "sparse"
                assert i.dimension is None
                assert i.deletion_protection == "disabled"
                break
            else:
                assert i.vector_type is not None

    async def test_sparse_index_deletion_protection(self, create_sl_index_params):
        pc = PineconeAsyncio()

        create_sl_index_params["metric"] = "dotproduct"
        create_sl_index_params["vector_type"] = "sparse"
        create_sl_index_params["deletion_protection"] = "enabled"
        del create_sl_index_params["dimension"]

        await pc.create_index(**create_sl_index_params)

        desc = await pc.describe_index(create_sl_index_params["name"])
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"
        assert desc.dimension is None
        assert desc.deletion_protection == "enabled"

        with pytest.raises(PineconeApiException) as e:
            await pc.delete_index(create_sl_index_params["name"], -1)
        assert "Deletion protection is enabled for this index" in str(e.value)

        await pc.configure_index(create_sl_index_params["name"], deletion_protection="disabled")

        desc2 = await pc.describe_index(create_sl_index_params["name"])
        assert desc2.deletion_protection == "disabled"

        await pc.delete_index(create_sl_index_params["name"], -1)


@pytest.mark.asyncio
class TestSparseIndexErrorCases:
    async def test_exception_when_passing_dimension(self, create_sl_index_params):
        pc = PineconeAsyncio()

        create_sl_index_params["metric"] = "dotproduct"
        create_sl_index_params["dimension"] = 10
        create_sl_index_params["vector_type"] = "sparse"

        with pytest.raises(ValueError) as e:
            await pc.create_index(**create_sl_index_params)
        assert "dimension should not be specified for sparse indexes" in str(e.value)

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    async def test_sparse_only_supports_dotproduct(self, create_sl_index_params, metric):
        pc = PineconeAsyncio()

        create_sl_index_params["metric"] = metric
        create_sl_index_params["vector_type"] = "sparse"
        del create_sl_index_params["dimension"]

        with pytest.raises(PineconeApiException) as e:
            await pc.create_index(**create_sl_index_params)
        assert "Sparse vector indexes must use the metric dotproduct." in str(e.value)
