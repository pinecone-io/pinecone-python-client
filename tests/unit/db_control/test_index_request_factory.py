import pytest
from pinecone import (
    ByocSpec,
    ServerlessSpec,
    PodSpec,
    CloudProvider,
    AwsRegion,
    PodType,
    PodIndexEnvironment,
    VectorType,
    Metric,
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


class TestTranslateLegacyRequest:
    """Tests for _translate_legacy_request method."""

    def test_translate_serverless_spec_to_deployment_and_schema_dense(self):
        """Test translating ServerlessSpec with dense vector to deployment + schema."""
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=1536, metric="cosine", vector_type="dense"
        )

        assert deployment == {
            "deployment_type": "serverless",
            "cloud": "aws",
            "region": "us-east-1",
        }
        assert schema == {
            "fields": {"_values": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
        }

    def test_translate_serverless_spec_to_deployment_and_schema_sparse(self):
        """Test translating ServerlessSpec with sparse vector to deployment + schema."""
        spec = ServerlessSpec(cloud="gcp", region="us-central1")
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, metric="dotproduct", vector_type="sparse"
        )

        assert deployment == {
            "deployment_type": "serverless",
            "cloud": "gcp",
            "region": "us-central1",
        }
        assert schema == {
            "fields": {"_sparse_values": {"type": "sparse_vector", "metric": "dotproduct"}}
        }

    def test_translate_pod_spec_to_deployment_and_schema(self):
        """Test translating PodSpec to deployment + schema."""
        spec = PodSpec(environment="us-west1-gcp", pod_type="p1.x1", replicas=2, shards=1, pods=2)
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=1024, metric="euclidean", vector_type="dense"
        )

        assert deployment == {
            "deployment_type": "pod",
            "environment": "us-west1-gcp",
            "pod_type": "p1.x1",
            "replicas": 2,
            "shards": 1,
            "pods": 2,
        }
        assert schema == {
            "fields": {
                "_values": {"type": "dense_vector", "dimension": 1024, "metric": "euclidean"}
            }
        }

    def test_translate_pod_spec_with_defaults(self):
        """Test translating PodSpec with default values."""
        spec = PodSpec(environment="us-east-1-aws")
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=768, metric="cosine", vector_type="dense"
        )

        assert deployment == {
            "deployment_type": "pod",
            "environment": "us-east-1-aws",
            "pod_type": "p1.x1",  # Default
            "replicas": 1,  # Default
            "shards": 1,  # Default
        }
        assert "pods" not in deployment  # Should not be included if None
        assert schema == {
            "fields": {"_values": {"type": "dense_vector", "dimension": 768, "metric": "cosine"}}
        }

    def test_translate_byoc_spec_to_deployment_and_schema(self):
        """Test translating ByocSpec to deployment + schema."""
        spec = ByocSpec(environment="aws-us-east-1-b921")
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=512, metric="dotproduct", vector_type="dense"
        )

        assert deployment == {"deployment_type": "byoc", "environment": "aws-us-east-1-b921"}
        assert schema == {
            "fields": {
                "_values": {"type": "dense_vector", "dimension": 512, "metric": "dotproduct"}
            }
        }

    def test_translate_serverless_spec_dict_to_deployment_and_schema(self):
        """Test translating ServerlessSpec as dict to deployment + schema."""
        spec = {"serverless": {"cloud": "aws", "region": "us-east-1"}}
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=1536, metric="cosine", vector_type="dense"
        )

        assert deployment == {
            "deployment_type": "serverless",
            "cloud": "aws",
            "region": "us-east-1",
        }
        assert schema == {
            "fields": {"_values": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
        }

    def test_translate_pod_spec_dict_to_deployment_and_schema(self):
        """Test translating PodSpec as dict to deployment + schema."""
        spec = {
            "pod": {"environment": "us-west1-gcp", "pod_type": "p1.x2", "replicas": 3, "shards": 2}
        }
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=2048, metric="cosine", vector_type="dense"
        )

        assert deployment == {
            "deployment_type": "pod",
            "environment": "us-west1-gcp",
            "pod_type": "p1.x2",
            "replicas": 3,
            "shards": 2,
        }
        assert schema == {
            "fields": {"_values": {"type": "dense_vector", "dimension": 2048, "metric": "cosine"}}
        }

    def test_translate_byoc_spec_dict_to_deployment_and_schema(self):
        """Test translating ByocSpec as dict to deployment + schema."""
        spec = {"byoc": {"environment": "gcp-us-central1-b123"}}
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=256, metric="euclidean", vector_type="dense"
        )

        assert deployment == {"deployment_type": "byoc", "environment": "gcp-us-central1-b123"}
        assert schema == {
            "fields": {"_values": {"type": "dense_vector", "dimension": 256, "metric": "euclidean"}}
        }

    def test_translate_sparse_vector_default_metric(self):
        """Test that sparse vector defaults to dotproduct metric."""
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, vector_type="sparse"
        )

        assert schema == {
            "fields": {
                "_sparse_values": {
                    "type": "sparse_vector",
                    "metric": "dotproduct",  # Default
                }
            }
        }

    def test_translate_dense_vector_default_metric(self):
        """Test that dense vector defaults to cosine metric."""
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=1536, vector_type="dense"
        )

        assert schema == {
            "fields": {
                "_values": {
                    "type": "dense_vector",
                    "dimension": 1536,
                    "metric": "cosine",  # Default
                }
            }
        }

    def test_translate_dense_vector_with_enum_metric(self):
        """Test translating with Metric enum."""
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=1536, metric=Metric.EUCLIDEAN, vector_type=VectorType.DENSE
        )

        assert schema == {
            "fields": {
                "_values": {"type": "dense_vector", "dimension": 1536, "metric": "euclidean"}
            }
        }

    def test_translate_dense_vector_requires_dimension(self):
        """Test that dense vector requires dimension."""
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        with pytest.raises(ValueError, match="dimension is required"):
            PineconeDBControlRequestFactory._translate_legacy_request(
                spec=spec, vector_type="dense"
            )

    def test_translate_invalid_spec_type(self):
        """Test that invalid spec type raises TypeError."""
        with pytest.raises(TypeError, match="spec must be of type"):
            PineconeDBControlRequestFactory._translate_legacy_request(
                spec="invalid", dimension=1536, vector_type="dense"
            )

    def test_translate_invalid_spec_dict(self):
        """Test that invalid spec dict raises ValueError."""
        with pytest.raises(ValueError, match="spec must contain"):
            PineconeDBControlRequestFactory._translate_legacy_request(
                spec={"invalid": {}}, dimension=1536, vector_type="dense"
            )

    def test_translate_dict_spec_with_enum_values(self):
        """Test that dict specs with enum values are converted to strings."""
        spec = {"serverless": {"cloud": CloudProvider.AWS, "region": AwsRegion.US_EAST_1}}
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=1536, metric="cosine", vector_type="dense"
        )

        assert deployment["cloud"] == "aws"  # Enum converted to string
        assert deployment["region"] == "us-east-1"  # Enum converted to string

    def test_translate_pod_spec_with_zero_replicas(self):
        """Test that zero replicas/shards are preserved (not converted to 1)."""
        spec = PodSpec(environment="us-east-1-aws", replicas=0, shards=0)
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=1536, metric="cosine", vector_type="dense"
        )

        assert deployment["replicas"] == 0  # Zero preserved
        assert deployment["shards"] == 0  # Zero preserved

    def test_translate_dict_spec_with_zero_replicas(self):
        """Test that zero replicas/shards in dict specs are preserved."""
        spec = {"pod": {"environment": "us-east-1-aws", "replicas": 0, "shards": 0}}
        deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=1536, metric="cosine", vector_type="dense"
        )

        assert deployment["replicas"] == 0  # Zero preserved
        assert deployment["shards"] == 0  # Zero preserved

    def test_translate_invalid_vector_type_raises_error(self):
        """Test that invalid vector_type raises ValueError instead of silently failing."""
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        with pytest.raises(ValueError, match="Invalid vector_type"):
            PineconeDBControlRequestFactory._translate_legacy_request(
                spec=spec, dimension=1536, vector_type="invalid_type"
            )

    def test_translate_invalid_vector_type_typo(self):
        """Test that typos in vector_type raise error."""
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        with pytest.raises(ValueError, match="Invalid vector_type"):
            PineconeDBControlRequestFactory._translate_legacy_request(
                spec=spec,
                dimension=1536,
                vector_type="desnse",  # Typo
            )
