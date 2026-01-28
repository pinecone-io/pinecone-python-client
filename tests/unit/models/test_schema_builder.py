"""Tests for SchemaBuilder fluent API."""

import os
import sys
import types

import pytest


def _load_schema_builder_module():
    """Load schema_builder.py as a standalone module to avoid broken imports."""
    # First load schema_fields (dependency)
    schema_fields_name = "pinecone.db_control.models.schema_fields"
    schema_fields = types.ModuleType(schema_fields_name)
    schema_fields.__file__ = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "pinecone",
        "db_control",
        "models",
        "schema_fields.py",
    )
    sys.modules[schema_fields_name] = schema_fields
    with open(schema_fields.__file__) as f:
        exec(compile(f.read(), schema_fields.__file__, "exec"), schema_fields.__dict__)

    # Now load schema_builder
    module_name = "pinecone.db_control.models.schema_builder"
    module = types.ModuleType(module_name)
    module.__file__ = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "pinecone",
        "db_control",
        "models",
        "schema_builder.py",
    )
    sys.modules[module_name] = module
    with open(module.__file__) as f:
        exec(compile(f.read(), module.__file__, "exec"), module.__dict__)

    return module


_schema_builder = _load_schema_builder_module()
SchemaBuilder = _schema_builder.SchemaBuilder


class TestSchemaBuilderBasic:
    def test_empty_builder_raises_on_build(self):
        builder = SchemaBuilder()
        with pytest.raises(ValueError, match="Cannot build empty schema"):
            builder.build()

    def test_single_text_field(self):
        schema = SchemaBuilder().text("title").build()
        assert schema == {"title": {"type": "string"}}

    def test_single_integer_field(self):
        schema = SchemaBuilder().integer("year").build()
        assert schema == {"year": {"type": "integer"}}

    def test_single_float_field(self):
        schema = SchemaBuilder().float("price").build()
        assert schema == {"price": {"type": "float"}}

    def test_single_dense_vector_field(self):
        schema = SchemaBuilder().dense_vector("embedding", dimension=1536, metric="cosine").build()
        assert schema == {
            "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}
        }

    def test_single_sparse_vector_field(self):
        schema = SchemaBuilder().sparse_vector("sparse").build()
        assert schema == {"sparse": {"type": "sparse_vector", "metric": "dotproduct"}}

    def test_single_semantic_text_field(self):
        schema = (
            SchemaBuilder()
            .semantic_text("content", model="multilingual-e5-large", field_map={"text": "content"})
            .build()
        )
        assert schema == {
            "content": {
                "type": "semantic_text",
                "model": "multilingual-e5-large",
                "field_map": {"text": "content"},
            }
        }


class TestSchemaBuilderChaining:
    def test_method_chaining_returns_builder(self):
        builder = SchemaBuilder()
        result = builder.text("title")
        assert result is builder

    def test_multiple_fields_chained(self):
        schema = (
            SchemaBuilder()
            .text("title", full_text_searchable=True)
            .integer("year", filterable=True)
            .dense_vector("embedding", dimension=1536, metric="cosine")
            .build()
        )
        assert schema == {
            "title": {"type": "string", "full_text_searchable": True},
            "year": {"type": "integer", "filterable": True},
            "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"},
        }

    def test_all_field_types_chained(self):
        schema = (
            SchemaBuilder()
            .text("title")
            .integer("year")
            .float("price")
            .dense_vector("dense", dimension=768, metric="euclidean")
            .sparse_vector("sparse")
            .semantic_text("content", model="model", field_map={"text": "content"})
            .build()
        )
        assert len(schema) == 6
        assert "title" in schema
        assert "year" in schema
        assert "price" in schema
        assert "dense" in schema
        assert "sparse" in schema
        assert "content" in schema


class TestSchemaBuilderFieldOptions:
    def test_text_with_all_options(self):
        schema = (
            SchemaBuilder()
            .text(
                "title",
                filterable=True,
                full_text_searchable=True,
                description="The document title",
            )
            .build()
        )
        assert schema == {
            "title": {
                "type": "string",
                "filterable": True,
                "full_text_searchable": True,
                "description": "The document title",
            }
        }

    def test_integer_with_all_options(self):
        schema = (
            SchemaBuilder().integer("year", filterable=True, description="Publication year").build()
        )
        assert schema == {
            "year": {"type": "integer", "filterable": True, "description": "Publication year"}
        }

    def test_float_with_all_options(self):
        schema = SchemaBuilder().float("price", filterable=True, description="Price in USD").build()
        assert schema == {
            "price": {"type": "float", "filterable": True, "description": "Price in USD"}
        }

    def test_dense_vector_with_description(self):
        schema = (
            SchemaBuilder()
            .dense_vector(
                "embedding", dimension=1536, metric="cosine", description="OpenAI embeddings"
            )
            .build()
        )
        assert schema == {
            "embedding": {
                "type": "dense_vector",
                "dimension": 1536,
                "metric": "cosine",
                "description": "OpenAI embeddings",
            }
        }

    def test_sparse_vector_with_options(self):
        schema = (
            SchemaBuilder()
            .sparse_vector("sparse", metric="dotproduct", description="BM25 vectors")
            .build()
        )
        assert schema == {
            "sparse": {
                "type": "sparse_vector",
                "metric": "dotproduct",
                "description": "BM25 vectors",
            }
        }

    def test_semantic_text_with_all_options(self):
        schema = (
            SchemaBuilder()
            .semantic_text(
                "content",
                model="multilingual-e5-large",
                field_map={"text": "content"},
                read_parameters={"truncate": "END"},
                write_parameters={"truncate": "START"},
                description="Semantic search field",
            )
            .build()
        )
        assert schema == {
            "content": {
                "type": "semantic_text",
                "model": "multilingual-e5-large",
                "field_map": {"text": "content"},
                "read_parameters": {"truncate": "END"},
                "write_parameters": {"truncate": "START"},
                "description": "Semantic search field",
            }
        }


class TestSchemaBuilderOverwrite:
    def test_adding_same_field_name_overwrites(self):
        schema = (
            SchemaBuilder()
            .text("field")
            .integer("field")  # Should overwrite the text field
            .build()
        )
        assert schema == {"field": {"type": "integer"}}


class TestSchemaBuilderUsageExamples:
    """Test the usage examples from the ticket."""

    def test_ticket_example(self):
        schema = (
            SchemaBuilder()
            .text("title", full_text_searchable=True)
            .integer("year", filterable=True)
            .dense_vector("embedding", dimension=1536, metric="cosine")
            .build()
        )
        assert schema == {
            "title": {"type": "string", "full_text_searchable": True},
            "year": {"type": "integer", "filterable": True},
            "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"},
        }
