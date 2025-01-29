import pytest
from pinecone import Metric, VectorType


class TestCreateSLIndexHappyPath:
    def test_create_index(self, client, create_sl_index_params):
        name = create_sl_index_params["name"]
        dimension = create_sl_index_params["dimension"]
        client.create_index(**create_sl_index_params)
        desc = client.describe_index(name)
        assert desc.name == name
        assert desc.dimension == dimension
        assert desc.metric == "cosine"
        assert desc.deletion_protection == "disabled"  # default value
        assert desc.vector_type == "dense"  # default value

    @pytest.mark.parametrize(
        "metric",
        ["cosine", "euclidean", "dotproduct", Metric.COSINE, Metric.EUCLIDEAN, Metric.DOTPRODUCT],
    )
    def test_create_default_index_with_metric(self, client, create_sl_index_params, metric):
        create_sl_index_params["metric"] = metric
        client.create_index(**create_sl_index_params)
        desc = client.describe_index(create_sl_index_params["name"])
        assert desc.metric == metric
        assert desc.vector_type == "dense"

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
