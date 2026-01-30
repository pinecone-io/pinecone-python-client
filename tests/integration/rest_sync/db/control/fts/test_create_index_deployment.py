"""Integration tests for index deployment configurations.

These tests verify the deployment parameter for schema-based index creation,
including explicit ServerlessDeployment and default deployment behavior.
"""

from pinecone import Pinecone, DenseVectorField, ServerlessDeployment


class TestCreateIndexWithDefaultDeployment:
    """Test creating indexes with default deployment (aws/us-east-1)."""

    def test_create_index_with_default_deployment(self, pc: Pinecone, index_name_and_tags):
        """Test that omitting deployment defaults to aws/us-east-1."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"


class TestCreateIndexWithExplicitDeployment:
    """Test creating indexes with explicit ServerlessDeployment."""

    def test_create_index_with_serverless_deployment_aws(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with explicit AWS ServerlessDeployment."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        resp = pc.db.index.create(
            name=index_name,
            schema=schema,
            deployment=ServerlessDeployment(cloud="aws", region="us-east-1"),
            tags=tags,
        )
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"

    def test_create_index_with_serverless_deployment_us_west(
        self, pc: Pinecone, index_name_and_tags
    ):
        """Test creating an index with AWS us-west-2 ServerlessDeployment."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        resp = pc.db.index.create(
            name=index_name,
            schema=schema,
            deployment=ServerlessDeployment(cloud="aws", region="us-west-2"),
            tags=tags,
        )
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-west-2"


class TestDeploymentCompatibilityAccess:
    """Test that deployment info is accessible via compatibility shim."""

    def test_spec_serverless_access_pattern(self, pc: Pinecone, index_name_and_tags):
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

        # Verify the compatibility shim provides .spec.serverless access
        assert desc.spec is not None
        assert desc.spec.serverless is not None
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"

        # Verify .spec.pod and .spec.byoc return None for serverless index
        assert desc.spec.pod is None
        assert desc.spec.byoc is None
