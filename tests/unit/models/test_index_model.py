from pinecone.db_control.models import IndexModel
from pinecone import CloudProvider, AwsRegion
from pinecone.config import OpenApiConfiguration

from tests.fixtures import make_index_model, make_index_status


class TestIndexModel:
    def test_serverless_index_model_with_dedicated_read_capacity(self):
        openapi_model = make_index_model(
            name="test-index-1",
            dimension=2,
            metric="cosine",
            host="https://test-index-1.pinecone.io",
            status=make_index_status(ready=True, state="Ready"),
            deletion_protection="enabled",
            spec={
                "serverless": {
                    "cloud": CloudProvider.AWS.value,
                    "region": AwsRegion.US_EAST_1.value,
                    "read_capacity": {
                        "mode": "Dedicated",
                        "status": {"state": "Ready"},
                        "dedicated": {
                            "node_type": "t1",
                            "scaling": "Manual",
                            "manual": {"shards": 1, "replicas": 1},
                        },
                    },
                }
            },
            _configuration=OpenApiConfiguration(),
        )

        wrapped = IndexModel(openapi_model)

        assert wrapped.name == "test-index-1"
        assert wrapped.dimension == 2
        assert wrapped.metric == "cosine"
        assert wrapped.host == "https://test-index-1.pinecone.io"
        assert wrapped.status.ready
        assert wrapped.status.state == "Ready"
        assert wrapped.deletion_protection == "enabled"

        assert wrapped.spec.serverless.read_capacity.mode == "Dedicated"
        assert wrapped.spec.serverless.read_capacity.dedicated.node_type == "t1"
        assert wrapped.spec.serverless.read_capacity.dedicated.scaling == "Manual"
        assert wrapped.spec.serverless.read_capacity.dedicated.manual.shards == 1
        assert wrapped.spec.serverless.read_capacity.dedicated.manual.replicas == 1
        assert wrapped["name"] == "test-index-1"

    def test_byoc_index_model_with_on_demand_read_capacity(self):
        """Test that BYOC indexes with OnDemand read_capacity deserialize correctly."""
        openapi_model = make_index_model(
            name="test-byoc-index",
            dimension=1536,
            metric="cosine",
            host="https://test-byoc-index.pinecone.io",
            status=make_index_status(ready=True, state="Ready"),
            deletion_protection="enabled",
            spec={
                "byoc": {
                    "environment": "us-east-1-aws",
                    "read_capacity": {"mode": "OnDemand", "status": {"state": "Ready"}},
                }
            },
            _configuration=OpenApiConfiguration(),
        )

        wrapped = IndexModel(openapi_model)

        assert wrapped.name == "test-byoc-index"
        assert wrapped.dimension == 1536
        assert wrapped.metric == "cosine"
        assert wrapped.host == "https://test-byoc-index.pinecone.io"
        assert wrapped.status.ready
        assert wrapped.status.state == "Ready"
        assert wrapped.deletion_protection == "enabled"
        assert wrapped.spec.byoc.environment == "us-east-1-aws"
        assert wrapped.spec.byoc.read_capacity.mode == "OnDemand"
        assert wrapped.spec.byoc.read_capacity.status.state == "Ready"
        assert wrapped["name"] == "test-byoc-index"

    def test_byoc_index_model_with_dedicated_read_capacity(self):
        """Test that BYOC indexes with Dedicated read_capacity deserialize correctly."""
        openapi_model = make_index_model(
            name="test-byoc-index-dedicated",
            dimension=1536,
            metric="cosine",
            host="https://test-byoc-index-dedicated.pinecone.io",
            status=make_index_status(ready=True, state="Ready"),
            deletion_protection="enabled",
            spec={
                "byoc": {
                    "environment": "us-east-1-aws",
                    "read_capacity": {
                        "mode": "Dedicated",
                        "status": {"state": "Ready"},
                        "dedicated": {
                            "node_type": "t1",
                            "scaling": "Manual",
                            "manual": {"shards": 2, "replicas": 3},
                        },
                    },
                }
            },
            _configuration=OpenApiConfiguration(),
        )

        wrapped = IndexModel(openapi_model)

        assert wrapped.name == "test-byoc-index-dedicated"
        assert wrapped.dimension == 1536
        assert wrapped.metric == "cosine"
        assert wrapped.host == "https://test-byoc-index-dedicated.pinecone.io"
        assert wrapped.status.ready
        assert wrapped.status.state == "Ready"
        assert wrapped.deletion_protection == "enabled"
        assert wrapped.spec.byoc.environment == "us-east-1-aws"
        assert wrapped.spec.byoc.read_capacity.mode == "Dedicated"
        assert wrapped.spec.byoc.read_capacity.status.state == "Ready"
        assert wrapped.spec.byoc.read_capacity.dedicated.node_type == "t1"
        assert wrapped.spec.byoc.read_capacity.dedicated.scaling == "Manual"
        assert wrapped.spec.byoc.read_capacity.dedicated.manual.shards == 2
        assert wrapped.spec.byoc.read_capacity.dedicated.manual.replicas == 3
        assert wrapped["name"] == "test-byoc-index-dedicated"
