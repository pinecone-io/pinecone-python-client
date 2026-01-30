"""Integration tests for describe_index compatibility.

These tests verify that describe_index returns compatible responses for
FTS indexes, with proper backward compatibility shims for the old API structure.
"""

from pinecone import Pinecone, TextField, DenseVectorField, SparseVectorField, ServerlessDeployment


class TestDescribeIndexVectorProperties:
    """Test that describe_index returns correct vector properties."""

    def test_describe_dense_vector_index(self, pc: Pinecone, index_name_and_tags):
        """Test that describe_index returns dimension, metric, vector_type for dense vectors."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=1536, metric="cosine")}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.name == index_name
        assert desc.dimension == 1536
        assert desc.metric == "cosine"
        assert desc.vector_type == "dense"

    def test_describe_dense_vector_index_euclidean(self, pc: Pinecone, index_name_and_tags):
        """Test that describe_index returns correct metric for euclidean distance."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=768, metric="euclidean")}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.name == index_name
        assert desc.dimension == 768
        assert desc.metric == "euclidean"
        assert desc.vector_type == "dense"

    def test_describe_dense_vector_index_dotproduct(self, pc: Pinecone, index_name_and_tags):
        """Test that describe_index returns correct metric for dotproduct."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="dotproduct")}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.name == index_name
        assert desc.dimension == 512
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "dense"

    def test_describe_sparse_vector_index(self, pc: Pinecone, index_name_and_tags):
        """Test that describe_index returns None dimension for sparse vectors."""
        index_name, tags = index_name_and_tags
        schema = {"sparse_embedding": SparseVectorField(metric="dotproduct")}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.name == index_name
        assert desc.dimension is None  # Sparse vectors don't have a fixed dimension
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"


class TestDescribeIndexSpecCompatibility:
    """Test the .spec compatibility shim for schema-based indexes."""

    def test_spec_serverless_cloud_access(self, pc: Pinecone, index_name_and_tags):
        """Test that .spec.serverless.cloud access pattern works."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        pc.db.index.create(
            name=index_name,
            schema=schema,
            deployment=ServerlessDeployment(cloud="aws", region="us-east-1"),
            tags=tags,
        )

        desc = pc.db.index.describe(name=index_name)

        # Old-style access via compatibility shim
        assert desc.spec is not None
        assert desc.spec.serverless is not None
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"

    def test_spec_pod_returns_none_for_serverless(self, pc: Pinecone, index_name_and_tags):
        """Test that .spec.pod returns None for serverless indexes."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.spec.pod is None

    def test_spec_byoc_returns_none_for_serverless(self, pc: Pinecone, index_name_and_tags):
        """Test that .spec.byoc returns None for serverless indexes."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.spec.byoc is None


class TestDescribeIndexWithMultipleFields:
    """Test describe_index for indexes with multiple fields."""

    def test_describe_index_with_text_and_vector(self, pc: Pinecone, index_name_and_tags):
        """Test describe_index with text and vector fields returns vector properties."""
        index_name, tags = index_name_and_tags
        schema = {
            "title": TextField(full_text_searchable=True),
            "category": TextField(filterable=True),
            "embedding": DenseVectorField(dimension=1024, metric="euclidean"),
        }
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.name == index_name
        # Vector properties should be extracted from the dense_vector field
        assert desc.dimension == 1024
        assert desc.metric == "euclidean"
        assert desc.vector_type == "dense"


class TestDescribeIndexStatus:
    """Test describe_index returns proper status information."""

    def test_describe_index_status_ready(self, pc: Pinecone, index_name_and_tags):
        """Test that describe_index returns ready status after creation."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        # Wait for index to be ready (default timeout)
        pc.db.index.create(name=index_name, schema=schema, tags=tags, timeout=None)

        desc = pc.db.index.describe(name=index_name)

        assert desc.status is not None
        assert desc.status.ready is True
        assert desc.status.state == "Ready"

    def test_describe_index_has_host(self, pc: Pinecone, index_name_and_tags):
        """Test that describe_index returns a valid host."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.host is not None
        assert isinstance(desc.host, str)
        assert desc.host != ""
        assert index_name in desc.host
