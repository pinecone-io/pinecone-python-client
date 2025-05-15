import pytest
from pinecone import (
    PineconeAsyncio,
    Metric,
    VectorType,
    DeletionProtection,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
)


@pytest.mark.asyncio
class TestAsyncioCreateIndex:
    @pytest.mark.parametrize("spec_fixture", ("spec1", "spec2", "spec3"))
    async def test_create_index(self, index_name, request, spec_fixture, index_tags):
        pc = PineconeAsyncio()
        spec = request.getfixturevalue(spec_fixture)

        resp = await pc.db.index.create(name=index_name, dimension=10, spec=spec, tags=index_tags)

        assert resp.name == index_name
        assert resp.dimension == 10
        assert resp.metric == "cosine"  # default value
        assert resp.vector_type == "dense"  # default value
        assert resp.deletion_protection == "disabled"  # default value

        desc = await pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 10
        assert desc.metric == "cosine"
        assert desc.deletion_protection == "disabled"  # default value
        assert desc.vector_type == "dense"  # default value
        await pc.close()

    async def test_create_skip_wait(self, index_name, spec1, index_tags):
        pc = PineconeAsyncio()
        resp = await pc.db.index.create(
            name=index_name, dimension=10, spec=spec1, timeout=-1, tags=index_tags
        )
        assert resp.name == index_name
        assert resp.dimension == 10
        assert resp.metric == "cosine"
        await pc.close()

    async def test_create_infinite_wait(self, index_name, spec1, index_tags):
        async with PineconeAsyncio() as pc:
            resp = await pc.db.index.create(
                name=index_name, dimension=10, spec=spec1, timeout=None, tags=index_tags
            )
            assert resp.name == index_name
            assert resp.dimension == 10
            assert resp.metric == "cosine"

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dotproduct"])
    async def test_create_default_index_with_metric(self, index_name, metric, spec1, index_tags):
        pc = PineconeAsyncio()

        await pc.db.index.create(
            name=index_name, dimension=10, spec=spec1, metric=metric, tags=index_tags
        )
        desc = await pc.db.index.describe(name=index_name)
        if isinstance(metric, str):
            assert desc.metric == metric
        else:
            assert desc.metric == metric.value
        assert desc.vector_type == "dense"
        await pc.close()

    @pytest.mark.parametrize(
        "metric_enum,vector_type_enum,dim",
        [
            (Metric.COSINE, VectorType.DENSE, 10),
            (Metric.EUCLIDEAN, VectorType.DENSE, 10),
            (Metric.DOTPRODUCT, VectorType.SPARSE, None),
        ],
    )
    async def test_create_with_enum_values_and_tags(
        self, index_name, metric_enum, vector_type_enum, dim, index_tags
    ):
        pc = PineconeAsyncio()
        args = {
            "name": index_name,
            "metric": metric_enum,
            "vector_type": vector_type_enum,
            "deletion_protection": DeletionProtection.DISABLED,
            "spec": ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            "tags": index_tags,
        }
        if dim is not None:
            args["dimension"] = dim

        await pc.db.index.create(**args)

        desc = await pc.db.index.describe(name=index_name)
        assert desc.metric == metric_enum.value
        assert desc.vector_type == vector_type_enum.value
        assert desc.dimension == dim
        assert desc.deletion_protection == DeletionProtection.DISABLED.value
        assert desc.name == index_name
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"
        assert desc.tags.to_dict() == index_tags
        await pc.db.index.delete(name=index_name)
        await pc.close()

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dotproduct"])
    async def test_create_dense_index_with_metric(self, index_name, spec1, metric, index_tags):
        pc = PineconeAsyncio()

        await pc.db.index.create(
            name=index_name,
            dimension=10,
            spec=spec1,
            metric=metric,
            vector_type=VectorType.DENSE,
            tags=index_tags,
        )

        desc = await pc.db.index.describe(name=index_name)
        assert desc.metric == metric
        assert desc.vector_type == "dense"
        await pc.close()

    async def test_create_with_optional_tags(self, index_name, spec1, index_tags):
        pc = PineconeAsyncio()

        await pc.db.index.create(name=index_name, dimension=10, spec=spec1, tags=index_tags)

        desc = await pc.db.index.describe(name=index_name)
        assert desc.tags.to_dict() == index_tags
        await pc.db.index.delete(name=index_name)
        await pc.close()

    async def test_create_sparse_index(self, index_name, spec1, index_tags):
        pc = PineconeAsyncio()

        await pc.db.index.create(
            name=index_name,
            spec=spec1,
            metric=Metric.DOTPRODUCT,
            vector_type=VectorType.SPARSE,
            tags=index_tags,
        )

        desc = await pc.db.index.describe(name=index_name)
        assert desc.vector_type == "sparse"
        assert desc.dimension is None
        assert desc.metric == "dotproduct"
        await pc.db.index.delete(name=index_name)
        await pc.close()

    async def test_create_with_deletion_protection(self, index_name, spec1, index_tags):
        pc = PineconeAsyncio()

        await pc.db.index.create(
            name=index_name,
            spec=spec1,
            metric=Metric.DOTPRODUCT,
            vector_type=VectorType.SPARSE,
            deletion_protection=DeletionProtection.ENABLED,
            tags=index_tags,
        )

        desc = await pc.db.index.describe(name=index_name)
        assert desc.deletion_protection == "enabled"
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"
        assert desc.dimension is None

        with pytest.raises(Exception):
            await pc.delete_index(index_name)

        await pc.configure_index(index_name, deletion_protection=DeletionProtection.DISABLED)

        desc2 = await pc.db.index.describe(name=index_name)
        assert desc2.deletion_protection == "disabled"
        await pc.close()
