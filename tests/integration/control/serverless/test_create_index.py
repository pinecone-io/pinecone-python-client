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


class TestCreateSLIndexWithReadCapacity:
    def test_create_index_with_ondemand_read_capacity(
        self, client: Pinecone, index_name, serverless_cloud, serverless_region
    ):
        """Test creating an index with explicit OnDemand read capacity."""
        resp = client.create_index(
            name=index_name,
            dimension=10,
            spec=ServerlessSpec(
                cloud=serverless_cloud,
                region=serverless_region,
                read_capacity={"mode": "OnDemand"},
            ),
        )
        assert resp.name == index_name
        assert resp.dimension == 10

        desc = client.describe_index(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 10
        assert desc.spec.serverless["cloud"] == serverless_cloud
        assert desc.spec.serverless["region"] == serverless_region
        # Verify read_capacity is set to OnDemand
        assert "read_capacity" in desc.spec.serverless
        assert desc.spec.serverless["read_capacity"]["mode"] == "OnDemand"

    def test_create_index_with_dedicated_read_capacity(
        self, client: Pinecone, index_name, serverless_cloud, serverless_region
    ):
        """Test creating an index with Dedicated read capacity configuration."""
        read_capacity_config = {
            "mode": "Dedicated",
            "dedicated": {
                "node_type": "t1",
                "scaling": "Manual",
                "manual": {
                    "shards": 2,
                    "replicas": 2,
                },
            },
        }

        resp = client.create_index(
            name=index_name,
            dimension=10,
            spec=ServerlessSpec(
                cloud=serverless_cloud,
                region=serverless_region,
                read_capacity=read_capacity_config,
            ),
        )
        assert resp.name == index_name
        assert resp.dimension == 10

        desc = client.describe_index(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 10
        assert desc.spec.serverless["cloud"] == serverless_cloud
        assert desc.spec.serverless["region"] == serverless_region
        # Verify read_capacity configuration
        assert "read_capacity" in desc.spec.serverless
        read_capacity = desc.spec.serverless["read_capacity"]
        assert read_capacity["mode"] == "Dedicated"
        assert "dedicated" in read_capacity
        assert read_capacity["dedicated"]["node_type"] == "t1"
        assert read_capacity["dedicated"]["scaling"] == "Manual"
        assert read_capacity["dedicated"]["manual"]["shards"] == 2
        assert read_capacity["dedicated"]["manual"]["replicas"] == 2

    def test_create_index_without_read_capacity_defaults_to_ondemand(
        self, client: Pinecone, index_name, serverless_cloud, serverless_region
    ):
        """Test that creating an index without read_capacity defaults to OnDemand."""
        resp = client.create_index(
            name=index_name,
            dimension=10,
            spec=ServerlessSpec(cloud=serverless_cloud, region=serverless_region),
        )
        assert resp.name == index_name

        desc = client.describe_index(name=index_name)
        # Verify read_capacity defaults to OnDemand (should be present in response)
        assert "read_capacity" in desc.spec.serverless
        # The API should return OnDemand as the default
        assert desc.spec.serverless["read_capacity"]["mode"] == "OnDemand"
