"""Integration tests for FTS-only indexes (text search without vectors).

These tests verify that indexes with only text fields (no vector fields)
return None for vector-related properties like dimension, metric, and vector_type.
"""

from pinecone import Pinecone, TextField, IntegerField, FloatField, SchemaBuilder


class TestFtsOnlyIndexCreation:
    """Test creating FTS-only indexes without vector fields."""

    def test_create_fts_only_index_single_text_field(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with only a single searchable text field."""
        index_name, tags = index_name_and_tags
        schema = {"content": TextField(full_text_searchable=True)}
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name

    def test_create_fts_only_index_multiple_text_fields(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with multiple text fields for FTS."""
        index_name, tags = index_name_and_tags
        schema = {
            "title": TextField(full_text_searchable=True),
            "body": TextField(full_text_searchable=True),
            "category": TextField(filterable=True),
        }
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

    def test_create_fts_only_index_with_numeric_fields(self, pc: Pinecone, index_name_and_tags):
        """Test creating an FTS-only index with text and numeric fields."""
        index_name, tags = index_name_and_tags
        schema = {
            "title": TextField(full_text_searchable=True),
            "year": IntegerField(filterable=True),
            "price": FloatField(filterable=True),
        }
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

    def test_create_fts_only_index_with_schema_builder(self, pc: Pinecone, index_name_and_tags):
        """Test creating an FTS-only index using SchemaBuilder."""
        index_name, tags = index_name_and_tags
        schema = (
            SchemaBuilder()
            .text("title", full_text_searchable=True)
            .text("description", full_text_searchable=True)
            .integer("year", filterable=True)
            .build()
        )
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name


class TestFtsOnlyIndexVectorProperties:
    """Test that FTS-only indexes return None for vector properties."""

    def test_fts_only_index_dimension_is_none(self, pc: Pinecone, index_name_and_tags):
        """Test that FTS-only indexes return None for dimension."""
        index_name, tags = index_name_and_tags
        schema = {"content": TextField(full_text_searchable=True)}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.name == index_name
        assert desc.dimension is None

    def test_fts_only_index_metric_is_none(self, pc: Pinecone, index_name_and_tags):
        """Test that FTS-only indexes return None for metric."""
        index_name, tags = index_name_and_tags
        schema = {"content": TextField(full_text_searchable=True)}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.name == index_name
        assert desc.metric is None

    def test_fts_only_index_vector_type_is_none(self, pc: Pinecone, index_name_and_tags):
        """Test that FTS-only indexes return None for vector_type."""
        index_name, tags = index_name_and_tags
        schema = {"content": TextField(full_text_searchable=True)}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.name == index_name
        assert desc.vector_type is None

    def test_fts_only_index_all_vector_properties_none(self, pc: Pinecone, index_name_and_tags):
        """Test that all vector properties are None for FTS-only indexes."""
        index_name, tags = index_name_and_tags
        schema = {
            "title": TextField(full_text_searchable=True),
            "category": TextField(filterable=True),
            "year": IntegerField(filterable=True),
        }
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        assert desc.name == index_name
        # All vector-related properties should be None
        assert desc.dimension is None
        assert desc.metric is None
        assert desc.vector_type is None


class TestFtsOnlyIndexSpecCompatibility:
    """Test spec compatibility shim for FTS-only indexes."""

    def test_fts_only_index_spec_serverless_access(self, pc: Pinecone, index_name_and_tags):
        """Test that .spec.serverless is accessible for FTS-only indexes."""
        index_name, tags = index_name_and_tags
        schema = {"content": TextField(full_text_searchable=True)}
        pc.db.index.create(name=index_name, schema=schema, tags=tags)

        desc = pc.db.index.describe(name=index_name)

        # Spec compatibility should still work
        assert desc.spec is not None
        assert desc.spec.serverless is not None
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"
