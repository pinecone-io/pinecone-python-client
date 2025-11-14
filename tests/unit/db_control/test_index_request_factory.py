import pytest
from pinecone import (
    ByocSpec,
    ServerlessSpec,
    PodSpec,
    CloudProvider,
    AwsRegion,
    PodType,
    PodIndexEnvironment,
)  # type: ignore[attr-defined]
from pinecone.db_control.request_factory import PineconeDBControlRequestFactory


class TestIndexRequestFactory:
    def test_create_index_request_with_spec_byoc(self):
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec=ByocSpec(environment="test-byoc-spec-id"),
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.byoc.environment == "test-byoc-spec-id"
        assert req.vector_type == "dense"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_serverless(self):
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.serverless.cloud == "aws"
        assert req.spec.serverless.region == "us-east-1"
        assert req.vector_type == "dense"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_serverless_dict(self):
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.serverless.cloud == "aws"
        assert req.spec.serverless.region == "us-east-1"
        assert req.vector_type == "dense"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_serverless_dict_enums(self):
        """Test that dict format with enum values is correctly converted to request body."""
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec={"serverless": {"cloud": CloudProvider.AWS, "region": AwsRegion.US_EAST_1}},
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.serverless.cloud == "aws"
        assert req.spec.serverless.region == "us-east-1"
        assert req.vector_type == "dense"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_byoc_dict(self):
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec={"byoc": {"environment": "test-byoc-spec-id"}},
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.byoc.environment == "test-byoc-spec-id"
        assert req.vector_type == "dense"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_pod(self):
        """Test creating index request with PodSpec object."""
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec=PodSpec(environment="us-west1-gcp", pod_type="p1.x1"),
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.pod.environment == "us-west1-gcp"
        assert req.spec.pod.pod_type == "p1.x1"
        assert req.vector_type == "dense"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_pod_all_fields(self):
        """Test creating index request with PodSpec object including all optional fields."""
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec=PodSpec(
                environment="us-west1-gcp",
                pod_type="p1.x1",
                pods=2,
                replicas=1,
                shards=1,
                metadata_config={"indexed": ["field1", "field2"]},
                source_collection="my-collection",
            ),
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.pod.environment == "us-west1-gcp"
        assert req.spec.pod.pod_type == "p1.x1"
        assert req.spec.pod.pods == 2
        assert req.spec.pod.replicas == 1
        assert req.spec.pod.shards == 1
        assert req.spec.pod.metadata_config.indexed == ["field1", "field2"]
        assert req.spec.pod.source_collection == "my-collection"
        assert req.vector_type == "dense"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_pod_dict(self):
        """Test creating index request with PodSpec as dictionary."""
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec={"pod": {"environment": "us-west1-gcp", "pod_type": "p1.x1"}},
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.pod.environment == "us-west1-gcp"
        assert req.spec.pod.pod_type == "p1.x1"
        assert req.vector_type == "dense"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_pod_dict_enums(self):
        """Test that dict format with enum values is correctly converted to request body."""
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec={
                "pod": {"environment": PodIndexEnvironment.US_WEST1_GCP, "pod_type": PodType.P1_X1}
            },
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.pod.environment == "us-west1-gcp"
        assert req.spec.pod.pod_type == "p1.x1"
        assert req.vector_type == "dense"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_pod_with_metadata_config(self):
        """Test creating index request with PodSpec including metadata_config."""
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec=PodSpec(
                environment="us-west1-gcp",
                pod_type="p1.x1",
                metadata_config={"indexed": ["genre", "year"]},
            ),
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.pod.environment == "us-west1-gcp"
        assert req.spec.pod.pod_type == "p1.x1"
        assert req.spec.pod.metadata_config.indexed == ["genre", "year"]
        assert req.vector_type == "dense"
        assert req.deletion_protection == "disabled"

    def test_parse_read_capacity_ondemand(self):
        """Test parsing OnDemand read capacity configuration."""
        read_capacity = {"mode": "OnDemand"}
        result = (
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        )
        assert result.mode == "OnDemand"

    def test_parse_read_capacity_dedicated_with_manual(self):
        """Test parsing Dedicated read capacity with manual scaling configuration."""
        read_capacity = {
            "mode": "Dedicated",
            "dedicated": {
                "node_type": "t1",
                "scaling": "Manual",
                "manual": {"shards": 2, "replicas": 3},
            },
        }
        result = (
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        )
        assert result.mode == "Dedicated"
        assert result.dedicated.node_type == "t1"
        assert result.dedicated.scaling == "Manual"
        assert result.dedicated.manual.shards == 2
        assert result.dedicated.manual.replicas == 3

    def test_parse_read_capacity_dedicated_missing_manual(self):
        """Test that missing manual configuration raises ValueError when scaling is Manual."""
        read_capacity = {"mode": "Dedicated", "dedicated": {"node_type": "t1", "scaling": "Manual"}}
        with pytest.raises(ValueError) as exc_info:
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        assert "manual" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()

    def test_parse_read_capacity_dedicated_missing_shards(self):
        """Test that missing shards in manual configuration raises ValueError."""
        read_capacity = {
            "mode": "Dedicated",
            "dedicated": {"node_type": "t1", "scaling": "Manual", "manual": {"replicas": 3}},
        }
        with pytest.raises(ValueError) as exc_info:
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        assert "shards" in str(exc_info.value).lower()

    def test_parse_read_capacity_dedicated_missing_replicas(self):
        """Test that missing replicas in manual configuration raises ValueError."""
        read_capacity = {
            "mode": "Dedicated",
            "dedicated": {"node_type": "t1", "scaling": "Manual", "manual": {"shards": 2}},
        }
        with pytest.raises(ValueError) as exc_info:
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        assert "replicas" in str(exc_info.value).lower()

    def test_parse_read_capacity_dedicated_missing_both_shards_and_replicas(self):
        """Test that missing both shards and replicas raises appropriate error."""
        read_capacity = {
            "mode": "Dedicated",
            "dedicated": {"node_type": "t1", "scaling": "Manual", "manual": {}},
        }
        with pytest.raises(ValueError) as exc_info:
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        assert "shards" in str(exc_info.value).lower()
        assert "replicas" in str(exc_info.value).lower()

    def test_parse_read_capacity_dedicated_invalid_manual_type(self):
        """Test that invalid manual type (not a dict) raises ValueError."""
        read_capacity = {
            "mode": "Dedicated",
            "dedicated": {"node_type": "t1", "scaling": "Manual", "manual": "invalid"},
        }
        with pytest.raises(ValueError) as exc_info:
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        assert "dictionary" in str(exc_info.value).lower()

    def test_parse_read_capacity_dedicated_missing_node_type(self):
        """Test that missing node_type raises ValueError."""
        read_capacity = {"mode": "Dedicated", "dedicated": {"scaling": "Manual"}}
        with pytest.raises(ValueError) as exc_info:
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        assert "node_type" in str(exc_info.value).lower()

    def test_parse_read_capacity_dedicated_missing_scaling(self):
        """Test that missing scaling raises ValueError."""
        read_capacity = {"mode": "Dedicated", "dedicated": {"node_type": "t1"}}
        with pytest.raises(ValueError) as exc_info:
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        assert "scaling" in str(exc_info.value).lower()
