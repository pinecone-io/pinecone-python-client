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


def _get_schema_field(req, field_name="_values"):
    """Helper to access schema fields from CreateIndexRequest."""
    schema = req.schema
    if hasattr(schema, "fields"):
        return schema.fields.get(field_name)
    return None


class TestIndexRequestFactory:
    def test_create_index_request_with_spec_byoc(self):
        """Test create_index_request translates legacy spec to schema+deployment format."""
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec=ByocSpec(environment="test-byoc-spec-id"),
        )
        assert req.name == "test-index"
        # In alpha API, metric/dimension are in schema.fields._values
        field = _get_schema_field(req)
        assert field is not None
        assert field.metric == "cosine"
        assert field.dimension == 1024
        assert field.type == "dense_vector"
        # deployment has deployment_type instead of spec
        assert req.deployment["deployment_type"] == "byoc"
        assert req.deployment["environment"] == "test-byoc-spec-id"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_serverless(self):
        """Test create_index_request with ServerlessSpec."""
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        assert req.name == "test-index"
        field = _get_schema_field(req)
        assert field is not None
        assert field.metric == "cosine"
        assert field.dimension == 1024
        assert req.deployment["deployment_type"] == "serverless"
        assert req.deployment["cloud"] == "aws"
        assert req.deployment["region"] == "us-east-1"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_serverless_dict(self):
        """Test create_index_request with serverless spec as dict."""
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )
        assert req.name == "test-index"
        field = _get_schema_field(req)
        assert field is not None
        assert field.metric == "cosine"
        assert field.dimension == 1024
        assert req.deployment["deployment_type"] == "serverless"
        assert req.deployment["cloud"] == "aws"
        assert req.deployment["region"] == "us-east-1"
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
        field = _get_schema_field(req)
        assert field is not None
        assert field.metric == "cosine"
        assert field.dimension == 1024
        assert req.deployment["deployment_type"] == "serverless"
        assert req.deployment["cloud"] == "aws"
        assert req.deployment["region"] == "us-east-1"
        assert req.deletion_protection == "disabled"

    def test_create_index_request_with_spec_byoc_dict(self):
        """Test create_index_request with byoc spec as dict."""
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec={"byoc": {"environment": "test-byoc-spec-id"}},
        )
        assert req.name == "test-index"
        field = _get_schema_field(req)
        assert field is not None
        assert field.metric == "cosine"
        assert field.dimension == 1024
        assert req.deployment["deployment_type"] == "byoc"
        assert req.deployment["environment"] == "test-byoc-spec-id"
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
        field = _get_schema_field(req)
        assert field is not None
        assert field.metric == "cosine"
        assert field.dimension == 1024
        assert req.deployment["deployment_type"] == "pod"
        assert req.deployment["environment"] == "us-west1-gcp"
        assert req.deployment["pod_type"] == "p1.x1"
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
        field = _get_schema_field(req)
        assert field is not None
        assert field.metric == "cosine"
        assert field.dimension == 1024
        assert req.deployment["deployment_type"] == "pod"
        assert req.deployment["environment"] == "us-west1-gcp"
        assert req.deployment["pod_type"] == "p1.x1"
        assert req.deployment["pods"] == 2
        assert req.deployment["replicas"] == 1
        assert req.deployment["shards"] == 1
        # Note: metadata_config and source_collection not yet supported in alpha deployment
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
        field = _get_schema_field(req)
        assert field is not None
        assert field.metric == "cosine"
        assert field.dimension == 1024
        assert req.deployment["deployment_type"] == "pod"
        assert req.deployment["environment"] == "us-west1-gcp"
        assert req.deployment["pod_type"] == "p1.x1"
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
        field = _get_schema_field(req)
        assert field is not None
        assert field.metric == "cosine"
        assert field.dimension == 1024
        assert req.deployment["deployment_type"] == "pod"
        assert req.deployment["environment"] == "us-west1-gcp"
        assert req.deployment["pod_type"] == "p1.x1"
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
        field = _get_schema_field(req)
        assert field is not None
        assert field.metric == "cosine"
        assert field.dimension == 1024
        assert req.deployment["deployment_type"] == "pod"
        assert req.deployment["environment"] == "us-west1-gcp"
        assert req.deployment["pod_type"] == "p1.x1"
        # Note: metadata_config not yet supported in alpha deployment
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
        """Test parsing Dedicated read capacity with manual scaling configuration.

        In alpha API, scaling is an object with strategy, replicas, and shards fields
        instead of a string with separate manual object.
        """
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
        # In alpha API, node_type and scaling are top-level on ReadCapacityDedicatedSpec
        assert result.node_type == "t1"
        # scaling is now an object with strategy, replicas, shards
        assert result.scaling.strategy == "Manual"
        assert result.scaling.shards == 2
        assert result.scaling.replicas == 3

    def test_parse_read_capacity_dedicated_with_defaults(self):
        """Test that missing shards/replicas default to 1 in alpha API."""
        read_capacity = {"mode": "Dedicated", "dedicated": {"node_type": "t1", "scaling": "Manual"}}
        result = (
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        )
        assert result.mode == "Dedicated"
        assert result.node_type == "t1"
        assert result.scaling.strategy == "Manual"
        # Alpha API defaults shards and replicas to 1
        assert result.scaling.shards == 1
        assert result.scaling.replicas == 1

    def test_parse_read_capacity_dedicated_partial_manual(self):
        """Test that partial manual config uses defaults for missing values."""
        read_capacity = {
            "mode": "Dedicated",
            "dedicated": {"node_type": "t1", "scaling": "Manual", "manual": {"replicas": 3}},
        }
        result = (
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        )
        assert result.scaling.replicas == 3
        assert result.scaling.shards == 1  # default

    def test_parse_read_capacity_dedicated_missing_node_type(self):
        """Test that missing node_type raises ValueError."""
        read_capacity = {"mode": "Dedicated", "dedicated": {"scaling": "Manual"}}
        with pytest.raises(ValueError) as exc_info:
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        assert "node_type" in str(exc_info.value).lower()

    def test_parse_read_capacity_dedicated_default_scaling(self):
        """Test that missing scaling defaults to Manual."""
        read_capacity = {"mode": "Dedicated", "dedicated": {"node_type": "t1"}}
        result = (
            PineconeDBControlRequestFactory._PineconeDBControlRequestFactory__parse_read_capacity(
                read_capacity
            )
        )
        assert result.mode == "Dedicated"
        assert result.node_type == "t1"
        # Alpha API defaults scaling strategy to "Manual"
        assert result.scaling.strategy == "Manual"


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


class TestTranslateEmbedToSemanticText:
    """Tests for _translate_embed_to_semantic_text method."""

    def test_basic_index_embed_translation(self):
        """Test basic IndexEmbed to semantic_text translation."""
        from pinecone.db_control.models import IndexEmbed

        embed = IndexEmbed(
            model="multilingual-e5-large", metric="cosine", field_map={"text": "synopsis"}
        )
        deployment, schema = PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
            cloud="aws", region="us-east-1", embed=embed
        )

        assert deployment == {
            "deployment_type": "serverless",
            "cloud": "aws",
            "region": "us-east-1",
        }
        assert schema == {
            "fields": {
                "synopsis": {
                    "type": "semantic_text",
                    "model": "multilingual-e5-large",
                    "metric": "cosine",
                    "read_parameters": {"input_type": "query"},
                    "write_parameters": {"input_type": "passage"},
                }
            }
        }

    def test_embed_translation_with_custom_parameters(self):
        """Test IndexEmbed translation with custom read/write parameters."""
        from pinecone.db_control.models import IndexEmbed

        embed = IndexEmbed(
            model="multilingual-e5-large",
            metric="dotproduct",
            field_map={"text": "content"},
            read_parameters={"input_type": "search_query", "truncate": "END"},
            write_parameters={"input_type": "search_document", "truncate": "END"},
        )
        deployment, schema = PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
            cloud="gcp", region="us-central1", embed=embed
        )

        assert deployment == {
            "deployment_type": "serverless",
            "cloud": "gcp",
            "region": "us-central1",
        }
        assert schema["fields"]["content"]["read_parameters"] == {
            "input_type": "search_query",
            "truncate": "END",
        }
        assert schema["fields"]["content"]["write_parameters"] == {
            "input_type": "search_document",
            "truncate": "END",
        }

    def test_embed_translation_without_metric(self):
        """Test IndexEmbed translation without metric (should not include metric in output)."""
        from pinecone.db_control.models import IndexEmbed

        embed = IndexEmbed(model="multilingual-e5-large", field_map={"text": "description"})
        deployment, schema = PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
            cloud="aws", region="us-west-2", embed=embed
        )

        assert "metric" not in schema["fields"]["description"]
        assert schema["fields"]["description"]["type"] == "semantic_text"
        assert schema["fields"]["description"]["model"] == "multilingual-e5-large"

    def test_embed_translation_with_dict(self):
        """Test dict-based embed configuration translation."""
        embed = {
            "model": "multilingual-e5-large",
            "metric": "euclidean",
            "field_map": {"text": "body"},
        }
        deployment, schema = PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
            cloud="aws", region="us-east-1", embed=embed
        )

        assert deployment["deployment_type"] == "serverless"
        assert schema["fields"]["body"]["type"] == "semantic_text"
        assert schema["fields"]["body"]["model"] == "multilingual-e5-large"
        assert schema["fields"]["body"]["metric"] == "euclidean"
        assert schema["fields"]["body"]["read_parameters"] == {"input_type": "query"}
        assert schema["fields"]["body"]["write_parameters"] == {"input_type": "passage"}

    def test_embed_translation_with_enum_cloud_region(self):
        """Test translation with enum values for cloud and region."""
        from pinecone.db_control.models import IndexEmbed

        embed = IndexEmbed(
            model="multilingual-e5-large", metric="cosine", field_map={"text": "synopsis"}
        )
        deployment, schema = PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
            cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1, embed=embed
        )

        assert deployment["cloud"] == "aws"
        assert deployment["region"] == "us-east-1"

    def test_embed_translation_multiple_field_mappings(self):
        """Test IndexEmbed translation with multiple field mappings."""
        from pinecone.db_control.models import IndexEmbed

        embed = IndexEmbed(
            model="multilingual-e5-large",
            metric="cosine",
            field_map={"text": "title", "description": "content"},
        )
        deployment, schema = PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
            cloud="aws", region="us-east-1", embed=embed
        )

        assert "title" in schema["fields"]
        assert "content" in schema["fields"]
        assert schema["fields"]["title"]["type"] == "semantic_text"
        assert schema["fields"]["content"]["type"] == "semantic_text"

    def test_embed_translation_missing_model_raises_error(self):
        """Test that missing model in dict embed raises ValueError."""
        embed = {"field_map": {"text": "synopsis"}}
        with pytest.raises(ValueError, match="model is required"):
            PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
                cloud="aws", region="us-east-1", embed=embed
            )

    def test_embed_translation_missing_field_map_raises_error(self):
        """Test that missing field_map in dict embed raises ValueError."""
        embed = {"model": "multilingual-e5-large"}
        with pytest.raises(ValueError, match="field_map is required"):
            PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
                cloud="aws", region="us-east-1", embed=embed
            )

    def test_embed_translation_empty_field_map_raises_error(self):
        """Test that empty field_map raises ValueError."""
        from pinecone.db_control.models import IndexEmbed

        embed = IndexEmbed(model="multilingual-e5-large", field_map={})
        with pytest.raises(ValueError, match="field_map must contain at least one mapping"):
            PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
                cloud="aws", region="us-east-1", embed=embed
            )

    def test_embed_translation_with_metric_enum(self):
        """Test IndexEmbed translation with Metric enum value."""
        from pinecone.db_control.models import IndexEmbed

        embed = IndexEmbed(
            model="multilingual-e5-large", metric=Metric.EUCLIDEAN, field_map={"text": "synopsis"}
        )
        deployment, schema = PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
            cloud="aws", region="us-east-1", embed=embed
        )

        assert schema["fields"]["synopsis"]["metric"] == "euclidean"

    def test_embed_translation_multiple_fields_independent_copies(self):
        """Test that multiple field mappings get independent copies of parameters."""
        from pinecone.db_control.models import IndexEmbed

        embed = IndexEmbed(
            model="multilingual-e5-large",
            metric="cosine",
            field_map={"text": "title", "description": "content"},
            read_parameters={"input_type": "search_query"},
            write_parameters={"input_type": "search_document"},
        )
        deployment, schema = PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
            cloud="aws", region="us-east-1", embed=embed
        )

        # Verify both fields have correct parameters
        assert schema["fields"]["title"]["read_parameters"] == {"input_type": "search_query"}
        assert schema["fields"]["content"]["read_parameters"] == {"input_type": "search_query"}

        # Verify dictionaries are independent copies (not shared references)
        assert (
            schema["fields"]["title"]["read_parameters"]
            is not schema["fields"]["content"]["read_parameters"]
        )
        assert (
            schema["fields"]["title"]["write_parameters"]
            is not schema["fields"]["content"]["write_parameters"]
        )

        # Verify modifying one doesn't affect the other
        schema["fields"]["title"]["read_parameters"]["extra"] = "value"
        assert "extra" not in schema["fields"]["content"]["read_parameters"]


class TestCreateIndexWithSchemaRequest:
    """Tests for create_index_with_schema_request method."""

    def test_basic_schema_creation(self):
        """Test basic schema-based index creation request."""
        from pinecone.db_control.models import DenseVectorField

        req = PineconeDBControlRequestFactory.create_index_with_schema_request(
            name="test-index",
            schema={"embedding": DenseVectorField(dimension=1536, metric="cosine")},
        )

        assert req.name == "test-index"
        assert req.deployment["deployment_type"] == "serverless"
        assert req.deployment["cloud"] == "aws"
        assert req.deployment["region"] == "us-east-1"
        assert req.schema["fields"]["embedding"]["type"] == "dense_vector"
        assert req.schema["fields"]["embedding"]["dimension"] == 1536
        assert req.schema["fields"]["embedding"]["metric"] == "cosine"

    def test_schema_with_dict_format(self):
        """Test schema creation with dict field definitions."""
        req = PineconeDBControlRequestFactory.create_index_with_schema_request(
            name="test-index",
            schema={
                "synopsis": {"type": "string", "full_text_searchable": True},
                "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"},
            },
        )

        assert req.name == "test-index"
        assert req.schema["fields"]["synopsis"]["type"] == "string"
        assert req.schema["fields"]["synopsis"]["full_text_searchable"] is True
        assert req.schema["fields"]["embedding"]["type"] == "dense_vector"

    def test_schema_with_custom_deployment(self):
        """Test schema creation with custom deployment configuration."""
        from pinecone.db_control.models import ServerlessDeployment, TextField

        deployment = ServerlessDeployment(cloud="gcp", region="us-central1")
        req = PineconeDBControlRequestFactory.create_index_with_schema_request(
            name="test-index",
            schema={"title": TextField(full_text_searchable=True)},
            deployment=deployment,
        )

        assert req.deployment["deployment_type"] == "serverless"
        assert req.deployment["cloud"] == "gcp"
        assert req.deployment["region"] == "us-central1"

    def test_schema_with_pod_deployment(self):
        """Test schema creation with pod deployment."""
        from pinecone.db_control.models import PodDeployment, DenseVectorField

        deployment = PodDeployment(
            environment="us-east-1-aws", pod_type="p1.x1", replicas=2, shards=1
        )
        req = PineconeDBControlRequestFactory.create_index_with_schema_request(
            name="test-index",
            schema={"embedding": DenseVectorField(dimension=1536, metric="cosine")},
            deployment=deployment,
        )

        assert req.deployment["deployment_type"] == "pod"
        assert req.deployment["environment"] == "us-east-1-aws"
        assert req.deployment["pod_type"] == "p1.x1"
        assert req.deployment["replicas"] == 2

    def test_schema_with_byoc_deployment(self):
        """Test schema creation with BYOC deployment."""
        from pinecone.db_control.models import ByocDeployment, DenseVectorField

        deployment = ByocDeployment(environment="my-byoc-env")
        req = PineconeDBControlRequestFactory.create_index_with_schema_request(
            name="test-index",
            schema={"embedding": DenseVectorField(dimension=1536, metric="cosine")},
            deployment=deployment,
        )

        assert req.deployment["deployment_type"] == "byoc"
        assert req.deployment["environment"] == "my-byoc-env"

    def test_schema_with_tags(self):
        """Test schema creation with tags."""
        from pinecone.db_control.models import DenseVectorField

        req = PineconeDBControlRequestFactory.create_index_with_schema_request(
            name="test-index",
            schema={"embedding": DenseVectorField(dimension=1536, metric="cosine")},
            tags={"env": "production", "team": "ml"},
        )

        assert req.tags["env"] == "production"
        assert req.tags["team"] == "ml"

    def test_schema_with_deletion_protection(self):
        """Test schema creation with deletion protection enabled."""
        from pinecone.db_control.models import DenseVectorField

        req = PineconeDBControlRequestFactory.create_index_with_schema_request(
            name="test-index",
            schema={"embedding": DenseVectorField(dimension=1536, metric="cosine")},
            deletion_protection="enabled",
        )

        assert req.deletion_protection == "enabled"

    def test_schema_with_multiple_fields(self):
        """Test schema creation with multiple field types."""
        from pinecone.db_control.models import TextField, IntegerField, DenseVectorField

        req = PineconeDBControlRequestFactory.create_index_with_schema_request(
            name="test-index",
            schema={
                "title": TextField(full_text_searchable=True),
                "year": IntegerField(filterable=True),
                "embedding": DenseVectorField(dimension=1536, metric="cosine"),
            },
        )

        assert req.schema["fields"]["title"]["type"] == "string"
        assert req.schema["fields"]["year"]["type"] == "integer"
        assert req.schema["fields"]["embedding"]["type"] == "dense_vector"

    def test_schema_invalid_field_type_raises_error(self):
        """Test that invalid field type raises TypeError."""
        with pytest.raises(TypeError, match="Invalid schema field type"):
            PineconeDBControlRequestFactory.create_index_with_schema_request(
                name="test-index", schema={"invalid": "not a field type"}
            )


class TestSerializeSchema:
    """Tests for _serialize_schema method."""

    def test_serialize_field_objects(self):
        """Test serializing field type objects to dicts."""
        from pinecone.db_control.models import TextField, DenseVectorField

        schema = {
            "title": TextField(full_text_searchable=True),
            "embedding": DenseVectorField(dimension=1536, metric="cosine"),
        }
        result = PineconeDBControlRequestFactory._serialize_schema(schema)

        assert result["fields"]["title"]["type"] == "string"
        assert result["fields"]["title"]["full_text_searchable"] is True
        assert result["fields"]["embedding"]["type"] == "dense_vector"
        assert result["fields"]["embedding"]["dimension"] == 1536

    def test_serialize_dict_format(self):
        """Test serializing dict field definitions."""
        schema = {
            "synopsis": {"type": "string", "full_text_searchable": True},
            "embedding": {"type": "dense_vector", "dimension": 1536},
        }
        result = PineconeDBControlRequestFactory._serialize_schema(schema)

        assert result["fields"]["synopsis"]["type"] == "string"
        assert result["fields"]["embedding"]["type"] == "dense_vector"
