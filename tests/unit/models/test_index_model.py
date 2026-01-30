from pinecone.core.openapi.db_control.models import (
    IndexModel as OpenApiIndexModel,
    IndexModelStatus,
)
from pinecone.core.openapi.db_control.model.schema import Schema
from pinecone.core.openapi.db_control.model.schema_fields import SchemaFields
from pinecone.core.openapi.db_control.model.deployment import Deployment
from pinecone.db_control.models import IndexModel
from pinecone.db_control.models.compatibility_spec import (
    CompatibilitySpec,
    ServerlessSpecCompat,
    PodSpecCompat,
    ByocSpecCompat,
)


class TestIndexModel:
    """Tests for IndexModel with the new alpha API format (2026-01.alpha)."""

    def test_index_model_serverless_deployment(self):
        """Test IndexModel with serverless deployment in alpha format."""
        schema = Schema(
            fields={
                "embedding": SchemaFields(type="dense_vector", dimension=1536, metric="cosine"),
                "title": SchemaFields(type="string", filterable=True),
            }
        )
        deployment = Deployment(deployment_type="serverless", cloud="aws", region="us-east-1")
        openapi_model = OpenApiIndexModel(
            name="test-alpha-index",
            schema=schema,
            deployment=deployment,
            host="https://test-alpha-index.pinecone.io",
            status=IndexModelStatus(ready=True, state="Ready"),
        )

        wrapped = IndexModel(openapi_model)

        # Test basic properties
        assert wrapped.name == "test-alpha-index"
        assert wrapped.host == "https://test-alpha-index.pinecone.io"

        # Test compatibility properties extracted from schema
        assert wrapped.dimension == 1536
        assert wrapped.metric == "cosine"
        assert wrapped.vector_type == "dense"

        # Test spec compatibility shim
        assert wrapped.spec is not None
        assert isinstance(wrapped.spec, CompatibilitySpec)
        assert wrapped.spec.serverless is not None
        assert isinstance(wrapped.spec.serverless, ServerlessSpecCompat)
        assert wrapped.spec.serverless.cloud == "aws"
        assert wrapped.spec.serverless.region == "us-east-1"
        assert wrapped.spec.pod is None
        assert wrapped.spec.byoc is None

    def test_index_model_pod_deployment(self):
        """Test IndexModel with pod deployment in alpha format."""
        schema = Schema(
            fields={
                "embedding": SchemaFields(type="dense_vector", dimension=768, metric="euclidean")
            }
        )
        deployment = Deployment(
            deployment_type="pod",
            environment="us-east-1-aws",
            pod_type="p1.x1",
            replicas=2,
            shards=1,
            pods=2,
        )
        openapi_model = OpenApiIndexModel(
            name="test-pod-index",
            schema=schema,
            deployment=deployment,
            host="https://test-pod-index.pinecone.io",
            status=IndexModelStatus(ready=True, state="Ready"),
        )

        wrapped = IndexModel(openapi_model)

        # Test compatibility properties
        assert wrapped.dimension == 768
        assert wrapped.metric == "euclidean"
        assert wrapped.vector_type == "dense"

        # Test spec compatibility shim
        assert wrapped.spec is not None
        assert wrapped.spec.pod is not None
        assert isinstance(wrapped.spec.pod, PodSpecCompat)
        assert wrapped.spec.pod.environment == "us-east-1-aws"
        assert wrapped.spec.pod.pod_type == "p1.x1"
        assert wrapped.spec.pod.replicas == 2
        assert wrapped.spec.pod.shards == 1
        assert wrapped.spec.pod.pods == 2
        assert wrapped.spec.serverless is None
        assert wrapped.spec.byoc is None

    def test_index_model_byoc_deployment(self):
        """Test IndexModel with BYOC deployment in alpha format."""
        schema = Schema(
            fields={
                "embedding": SchemaFields(type="dense_vector", dimension=512, metric="dotproduct")
            }
        )
        deployment = Deployment(deployment_type="byoc", environment="aws-us-east-1-b92")
        openapi_model = OpenApiIndexModel(
            name="test-byoc-index",
            schema=schema,
            deployment=deployment,
            host="https://test-byoc-index.pinecone.io",
            status=IndexModelStatus(ready=True, state="Ready"),
        )

        wrapped = IndexModel(openapi_model)

        # Test compatibility properties
        assert wrapped.dimension == 512
        assert wrapped.metric == "dotproduct"
        assert wrapped.vector_type == "dense"

        # Test spec compatibility shim
        assert wrapped.spec is not None
        assert wrapped.spec.byoc is not None
        assert isinstance(wrapped.spec.byoc, ByocSpecCompat)
        assert wrapped.spec.byoc.environment == "aws-us-east-1-b92"
        assert wrapped.spec.serverless is None
        assert wrapped.spec.pod is None

    def test_index_model_sparse_vector(self):
        """Test IndexModel with sparse vector field."""
        schema = Schema(
            fields={"sparse_embedding": SchemaFields(type="sparse_vector", metric="dotproduct")}
        )
        deployment = Deployment(deployment_type="serverless", cloud="aws", region="us-east-1")
        openapi_model = OpenApiIndexModel(
            name="test-sparse-index",
            schema=schema,
            deployment=deployment,
            host="https://test-sparse-index.pinecone.io",
            status=IndexModelStatus(ready=True, state="Ready"),
        )

        wrapped = IndexModel(openapi_model)

        # Sparse vectors don't have dimension
        assert wrapped.dimension is None
        assert wrapped.metric == "dotproduct"
        assert wrapped.vector_type == "sparse"

    def test_index_model_fts_only(self):
        """Test IndexModel with FTS-only index (no vector fields)."""
        schema = Schema(
            fields={
                "title": SchemaFields(type="string", full_text_searchable=True),
                "content": SchemaFields(type="string", full_text_searchable=True),
                "year": SchemaFields(type="integer", filterable=True),
            }
        )
        deployment = Deployment(deployment_type="serverless", cloud="aws", region="us-east-1")
        openapi_model = OpenApiIndexModel(
            name="test-fts-index",
            schema=schema,
            deployment=deployment,
            host="https://test-fts-index.pinecone.io",
            status=IndexModelStatus(ready=True, state="Ready"),
        )

        wrapped = IndexModel(openapi_model)

        # FTS-only indexes return None for vector properties
        assert wrapped.dimension is None
        assert wrapped.metric is None
        assert wrapped.vector_type is None

        # spec still works
        assert wrapped.spec is not None
        assert wrapped.spec.serverless is not None

    def test_index_model_dict_access(self):
        """Test dict-style access to IndexModel properties."""
        schema = Schema(
            fields={"embedding": SchemaFields(type="dense_vector", dimension=1024, metric="cosine")}
        )
        deployment = Deployment(deployment_type="serverless", cloud="gcp", region="us-central1")
        openapi_model = OpenApiIndexModel(
            name="test-dict-access",
            schema=schema,
            deployment=deployment,
            host="https://test-dict-access.pinecone.io",
            status=IndexModelStatus(ready=True, state="Ready"),
        )

        wrapped = IndexModel(openapi_model)

        # Test dict-style access
        assert wrapped["name"] == "test-dict-access"
        assert wrapped["dimension"] == 1024
        assert wrapped["metric"] == "cosine"
        assert wrapped["vector_type"] == "dense"


class TestCompatibilitySpec:
    """Direct tests for CompatibilitySpec class."""

    def test_serverless_spec_compat(self):
        """Test ServerlessSpecCompat dataclass."""
        spec = ServerlessSpecCompat(cloud="aws", region="us-west-2")
        assert spec.cloud == "aws"
        assert spec.region == "us-west-2"

    def test_pod_spec_compat(self):
        """Test PodSpecCompat dataclass."""
        spec = PodSpecCompat(
            environment="us-east-1-aws",
            pod_type="s1.x2",
            replicas=3,
            shards=2,
            pods=6,
            metadata_config={"indexed": ["field1"]},
            source_collection="my-collection",
        )
        assert spec.environment == "us-east-1-aws"
        assert spec.pod_type == "s1.x2"
        assert spec.replicas == 3
        assert spec.shards == 2
        assert spec.pods == 6
        assert spec.metadata_config == {"indexed": ["field1"]}
        assert spec.source_collection == "my-collection"

    def test_byoc_spec_compat(self):
        """Test ByocSpecCompat dataclass."""
        spec = ByocSpecCompat(environment="my-byoc-env")
        assert spec.environment == "my-byoc-env"

    def test_compatibility_spec_serverless(self):
        """Test CompatibilitySpec with serverless deployment."""

        class MockDeployment:
            deployment_type = "serverless"
            cloud = "azure"
            region = "eastus"

        compat = CompatibilitySpec(MockDeployment())
        assert compat.serverless is not None
        assert compat.serverless.cloud == "azure"
        assert compat.serverless.region == "eastus"
        assert compat.pod is None
        assert compat.byoc is None

    def test_compatibility_spec_pod(self):
        """Test CompatibilitySpec with pod deployment."""

        class MockDeployment:
            deployment_type = "pod"
            environment = "us-west-2-aws"
            pod_type = "p2.x1"
            replicas = 1
            shards = 1
            pods = 1
            metadata_config = None
            source_collection = None

        compat = CompatibilitySpec(MockDeployment())
        assert compat.pod is not None
        assert compat.pod.environment == "us-west-2-aws"
        assert compat.pod.pod_type == "p2.x1"
        assert compat.serverless is None
        assert compat.byoc is None

    def test_compatibility_spec_byoc(self):
        """Test CompatibilitySpec with BYOC deployment."""

        class MockDeployment:
            deployment_type = "byoc"
            environment = "custom-env"

        compat = CompatibilitySpec(MockDeployment())
        assert compat.byoc is not None
        assert compat.byoc.environment == "custom-env"
        assert compat.serverless is None
        assert compat.pod is None
