"""Integration tests for schema-based index creation.

These tests verify the new schema-based API for creating Pinecone indexes
with full-text search and vector capabilities.
"""

from pinecone import (
    Pinecone,
    TextField,
    IntegerField,
    FloatField,
    DenseVectorField,
    SparseVectorField,
    SchemaBuilder,
    DeletionProtection,
)


class TestCreateIndexWithSchemaDict:
    """Test creating indexes with schema as a dictionary of raw dicts."""

    def test_create_index_with_dense_vector_dict(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with a dense vector field using dict format."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 1536
        assert desc.metric == "cosine"
        assert desc.vector_type == "dense"

    def test_create_index_with_text_and_vector_dict(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with text and vector fields using dict format."""
        index_name, tags = index_name_and_tags
        schema = {
            "title": {"type": "string", "full_text_searchable": True},
            "category": {"type": "string", "filterable": True},
            "embedding": {"type": "dense_vector", "dimension": 1024, "metric": "euclidean"},
        }
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 1024
        assert desc.metric == "euclidean"
        assert desc.vector_type == "dense"

    def test_create_index_with_sparse_vector_dict(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with a sparse vector field using dict format."""
        index_name, tags = index_name_and_tags
        schema = {"sparse_embedding": {"type": "sparse_vector", "metric": "dotproduct"}}
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"

    def test_create_index_with_numeric_fields_dict(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with integer and float fields using dict format."""
        index_name, tags = index_name_and_tags
        schema = {
            "year": {"type": "integer", "filterable": True},
            "price": {"type": "float", "filterable": True},
            "embedding": {"type": "dense_vector", "dimension": 512, "metric": "dotproduct"},
        }
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 512
        assert desc.metric == "dotproduct"


class TestCreateIndexWithFieldClasses:
    """Test creating indexes using typed field classes."""

    def test_create_index_with_text_field(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with TextField class."""
        index_name, tags = index_name_and_tags
        schema = {
            "title": TextField(full_text_searchable=True),
            "embedding": DenseVectorField(dimension=768, metric="cosine"),
        }
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 768
        assert desc.metric == "cosine"
        assert desc.vector_type == "dense"

    def test_create_index_with_integer_field(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with IntegerField class."""
        index_name, tags = index_name_and_tags
        schema = {
            "count": IntegerField(filterable=True),
            "embedding": DenseVectorField(dimension=256, metric="euclidean"),
        }
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.dimension == 256

    def test_create_index_with_float_field(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with FloatField class."""
        index_name, tags = index_name_and_tags
        schema = {
            "score": FloatField(filterable=True),
            "embedding": DenseVectorField(dimension=384, metric="cosine"),
        }
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.dimension == 384

    def test_create_index_with_dense_vector_field(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with DenseVectorField class."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=1024, metric="dotproduct")}
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 1024
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "dense"

    def test_create_index_with_sparse_vector_field(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with SparseVectorField class."""
        index_name, tags = index_name_and_tags
        schema = {"sparse_embedding": SparseVectorField(metric="dotproduct")}
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"

    def test_create_index_with_field_descriptions(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with field descriptions."""
        index_name, tags = index_name_and_tags
        schema = {
            "title": TextField(full_text_searchable=True, description="The document title"),
            "year": IntegerField(filterable=True, description="Publication year"),
            "embedding": DenseVectorField(
                dimension=512, metric="cosine", description="Document embedding"
            ),
        }
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name


class TestCreateIndexWithSchemaBuilder:
    """Test creating indexes using SchemaBuilder fluent API."""

    def test_create_index_with_schema_builder_text_and_vector(
        self, pc: Pinecone, index_name_and_tags
    ):
        """Test creating an index using SchemaBuilder with text and vector fields."""
        index_name, tags = index_name_and_tags
        schema = (
            SchemaBuilder()
            .text("title", full_text_searchable=True)
            .dense_vector("embedding", dimension=1536, metric="cosine")
            .build()
        )
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 1536
        assert desc.metric == "cosine"

    def test_create_index_with_schema_builder_multiple_fields(
        self, pc: Pinecone, index_name_and_tags
    ):
        """Test creating an index using SchemaBuilder with multiple field types."""
        index_name, tags = index_name_and_tags
        schema = (
            SchemaBuilder()
            .text("title", full_text_searchable=True)
            .text("category", filterable=True)
            .integer("year", filterable=True)
            .float("score", filterable=True)
            .dense_vector("embedding", dimension=768, metric="euclidean")
            .build()
        )
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 768
        assert desc.metric == "euclidean"

    def test_create_index_with_schema_builder_sparse_vector(
        self, pc: Pinecone, index_name_and_tags
    ):
        """Test creating an index using SchemaBuilder with sparse vector field."""
        index_name, tags = index_name_and_tags
        schema = (
            SchemaBuilder()
            .text("content", full_text_searchable=True)
            .sparse_vector("sparse_embedding")
            .build()
        )
        resp = pc.db.index.create(name=index_name, schema=schema, tags=tags)
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"


class TestCreateIndexWithDeletionProtection:
    """Test creating indexes with deletion protection enabled."""

    def test_create_index_with_deletion_protection_enabled(self, pc: Pinecone, index_name_and_tags):
        """Test creating an index with deletion protection enabled."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        resp = pc.db.index.create(
            name=index_name,
            schema=schema,
            deletion_protection=DeletionProtection.ENABLED,
            tags=tags,
        )
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.deletion_protection == "enabled"

        # Disable deletion protection for cleanup
        pc.db.index.configure(name=index_name, deletion_protection=DeletionProtection.DISABLED)

    def test_create_index_with_deletion_protection_disabled(
        self, pc: Pinecone, index_name_and_tags
    ):
        """Test creating an index with deletion protection explicitly disabled."""
        index_name, tags = index_name_and_tags
        schema = {"embedding": DenseVectorField(dimension=512, metric="cosine")}
        resp = pc.db.index.create(
            name=index_name,
            schema=schema,
            deletion_protection=DeletionProtection.DISABLED,
            tags=tags,
        )
        assert resp.name == index_name

        desc = pc.db.index.describe(name=index_name)
        assert desc.deletion_protection == "disabled"
