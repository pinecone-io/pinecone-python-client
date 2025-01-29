import pytest
from pinecone import (
    Pinecone,
    Metric,
    VectorType,
    DeletionProtection,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
)


class TestCreateSLIndexHappyPath:
    def test_create_index(self, client: Pinecone, index_name):
        resp = client.create_index(
            name=index_name,
            dimension=10,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
        )
        assert resp.name == index_name
        assert resp.dimension == 10
        assert resp.metric == "cosine"  # default value
        assert resp.vector_type == "dense"  # default value
        assert resp.deletion_protection == "disabled"  # default value

        desc = client.describe_index(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 10
        assert desc.metric == "cosine"
        assert desc.deletion_protection == "disabled"  # default value
        assert desc.vector_type == "dense"  # default value

    def test_create_skip_wait(self, client, index_name):
        resp = client.create_index(
            name=index_name,
            dimension=10,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            timeout=-1,
        )
        assert resp.name == index_name
        assert resp.dimension == 10
        assert resp.metric == "cosine"

    def test_create_infinite_wait(self, client, index_name):
        resp = client.create_index(
            name=index_name,
            dimension=10,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            timeout=None,
        )
        assert resp.name == index_name
        assert resp.dimension == 10
        assert resp.metric == "cosine"

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dotproduct"])
    def test_create_default_index_with_metric(self, client, create_sl_index_params, metric):
        create_sl_index_params["metric"] = metric
        client.create_index(**create_sl_index_params)
        desc = client.describe_index(create_sl_index_params["name"])
        if isinstance(metric, str):
            assert desc.metric == metric
        else:
            assert desc.metric == metric.value
        assert desc.vector_type == "dense"

    @pytest.mark.parametrize(
        "metric_enum,vector_type_enum,dim,tags",
        [
            (Metric.COSINE, VectorType.DENSE, 10, None),
            (Metric.EUCLIDEAN, VectorType.DENSE, 10, {"env": "prod"}),
            (Metric.DOTPRODUCT, VectorType.SPARSE, None, {"env": "dev"}),
        ],
    )
    def test_create_with_enum_values(
        self, client, index_name, metric_enum, vector_type_enum, dim, tags
    ):
        args = {
            "name": index_name,
            "metric": metric_enum,
            "vector_type": vector_type_enum,
            "deletion_protection": DeletionProtection.DISABLED,
            "spec": ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            "tags": tags,
        }
        if dim is not None:
            args["dimension"] = dim

        client.create_index(**args)

        desc = client.describe_index(index_name)
        assert desc.metric == metric_enum.value
        assert desc.vector_type == vector_type_enum.value
        assert desc.dimension == dim
        assert desc.deletion_protection == DeletionProtection.DISABLED.value
        assert desc.name == index_name
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"
        if tags:
            assert desc.tags.to_dict() == tags

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dotproduct"])
    def test_create_dense_index_with_metric(self, client, create_sl_index_params, metric):
        create_sl_index_params["metric"] = metric
        create_sl_index_params["vector_type"] = VectorType.DENSE
        client.create_index(**create_sl_index_params)
        desc = client.describe_index(create_sl_index_params["name"])
        assert desc.metric == metric
        assert desc.vector_type == "dense"

    def test_create_with_optional_tags(self, client, create_sl_index_params):
        tags = {"foo": "FOO", "bar": "BAR"}
        create_sl_index_params["tags"] = tags
        client.create_index(**create_sl_index_params)
        desc = client.describe_index(create_sl_index_params["name"])
        assert desc.tags.to_dict() == tags
