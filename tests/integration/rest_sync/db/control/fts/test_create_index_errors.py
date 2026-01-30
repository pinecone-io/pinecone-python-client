"""Integration tests for schema-based index creation error handling.

These tests verify proper error handling for invalid configurations
and mutual exclusion of spec and schema parameters.
"""

import pytest
from pinecone import Pinecone, DenseVectorField, ServerlessSpec, CloudProvider, AwsRegion


class TestSpecAndSchemaMutualExclusion:
    """Test that spec and schema parameters are mutually exclusive."""

    def test_error_when_both_spec_and_schema_provided(self, pc: Pinecone, index_name_and_tags):
        """Test that providing both spec and schema raises ValueError."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        spec = ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1)

        with pytest.raises(ValueError) as exc_info:
            pc.db.index.create(name=index_name, spec=spec, schema=schema, tags=tags)

        assert "spec" in str(exc_info.value).lower()
        assert "schema" in str(exc_info.value).lower()

    def test_error_when_neither_spec_nor_schema_provided(self, pc: Pinecone, index_name_and_tags):
        """Test that providing neither spec nor schema raises ValueError."""
        index_name, tags = index_name_and_tags

        with pytest.raises(ValueError) as exc_info:
            pc.db.index.create(name=index_name, dimension=512, metric="cosine", tags=tags)

        assert "spec" in str(exc_info.value).lower() or "schema" in str(exc_info.value).lower()


class TestSchemaValidation:
    """Test schema field configuration validation."""

    def test_error_when_schema_is_empty(self, pc: Pinecone, index_name_and_tags):
        """Test that an empty schema raises an error from the API."""
        index_name, tags = index_name_and_tags
        schema = {}

        with pytest.raises(Exception):
            pc.db.index.create(name=index_name, schema=schema, tags=tags)


class TestLegacySpecBasedCreation:
    """Test that legacy spec-based creation still works correctly."""

    def test_legacy_serverless_spec_creation(self, pc: Pinecone, index_name_and_tags):
        """Test that legacy ServerlessSpec-based creation still works."""
        index_name, tags = index_name_and_tags
        resp = pc.db.index.create(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            tags=tags,
        )
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 1024
        assert desc.metric == "cosine"
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"

    def test_legacy_dict_spec_creation(self, pc: Pinecone, index_name_and_tags):
        """Test that legacy dict-based spec creation still works."""
        index_name, tags = index_name_and_tags
        resp = pc.db.index.create(
            name=index_name,
            dimension=768,
            metric="euclidean",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            tags=tags,
        )
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 768
        assert desc.metric == "euclidean"

    def test_legacy_vector_type_parameter(self, pc: Pinecone, index_name_and_tags):
        """Test that legacy vector_type parameter works with spec."""
        index_name, tags = index_name_and_tags
        resp = pc.db.index.create(
            name=index_name,
            dimension=512,
            metric="cosine",
            vector_type="dense",
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            tags=tags,
        )
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.vector_type == "dense"

    def test_legacy_sparse_index_creation(self, pc: Pinecone, index_name_and_tags):
        """Test that legacy sparse index creation still works."""
        index_name, tags = index_name_and_tags
        resp = pc.db.index.create(
            name=index_name,
            metric="dotproduct",
            vector_type="sparse",
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            tags=tags,
        )
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"
        assert desc.dimension is None  # Sparse indexes have no dimension
