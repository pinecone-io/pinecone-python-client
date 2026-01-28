"""Tests for schema field type classes.

Note: We import the schema_fields module by creating a standalone module
since the alpha API changes have broken the normal import chain through
db_control. Once SDK-104/107 are complete, these tests can be updated
to import from the normal locations.
"""

import os
import sys
import types


def _load_schema_fields_module():
    """Load schema_fields.py as a standalone module to avoid broken imports."""
    module_name = "pinecone.db_control.models.schema_fields"

    # Create module and register it before exec
    module = types.ModuleType(module_name)
    module.__file__ = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "pinecone",
        "db_control",
        "models",
        "schema_fields.py",
    )
    sys.modules[module_name] = module

    # Execute the module code
    with open(module.__file__) as f:
        code = compile(f.read(), module.__file__, "exec")
        exec(code, module.__dict__)

    return module


_schema_fields = _load_schema_fields_module()

TextField = _schema_fields.TextField
IntegerField = _schema_fields.IntegerField
FloatField = _schema_fields.FloatField
DenseVectorField = _schema_fields.DenseVectorField
SparseVectorField = _schema_fields.SparseVectorField
SemanticTextField = _schema_fields.SemanticTextField


class TestTextField:
    def test_default_values(self):
        field = TextField()
        assert field.filterable is False
        assert field.full_text_searchable is False
        assert field.description is None

    def test_to_dict_minimal(self):
        field = TextField()
        result = field.to_dict()
        assert result == {"type": "string"}

    def test_to_dict_filterable(self):
        field = TextField(filterable=True)
        result = field.to_dict()
        assert result == {"type": "string", "filterable": True}

    def test_to_dict_full_text_searchable(self):
        field = TextField(full_text_searchable=True)
        result = field.to_dict()
        assert result == {"type": "string", "full_text_searchable": True}

    def test_to_dict_all_options(self):
        field = TextField(filterable=True, full_text_searchable=True, description="A text field")
        result = field.to_dict()
        assert result == {
            "type": "string",
            "filterable": True,
            "full_text_searchable": True,
            "description": "A text field",
        }


class TestIntegerField:
    def test_default_values(self):
        field = IntegerField()
        assert field.filterable is False
        assert field.description is None

    def test_to_dict_minimal(self):
        field = IntegerField()
        result = field.to_dict()
        assert result == {"type": "integer"}

    def test_to_dict_filterable(self):
        field = IntegerField(filterable=True)
        result = field.to_dict()
        assert result == {"type": "integer", "filterable": True}

    def test_to_dict_with_description(self):
        field = IntegerField(filterable=True, description="Year of publication")
        result = field.to_dict()
        assert result == {
            "type": "integer",
            "filterable": True,
            "description": "Year of publication",
        }


class TestFloatField:
    def test_default_values(self):
        field = FloatField()
        assert field.filterable is False
        assert field.description is None

    def test_to_dict_minimal(self):
        field = FloatField()
        result = field.to_dict()
        assert result == {"type": "float"}

    def test_to_dict_filterable(self):
        field = FloatField(filterable=True)
        result = field.to_dict()
        assert result == {"type": "float", "filterable": True}

    def test_to_dict_with_description(self):
        field = FloatField(filterable=True, description="Price in USD")
        result = field.to_dict()
        assert result == {"type": "float", "filterable": True, "description": "Price in USD"}


class TestDenseVectorField:
    def test_required_params(self):
        field = DenseVectorField(dimension=1536, metric="cosine")
        assert field.dimension == 1536
        assert field.metric == "cosine"
        assert field.description is None

    def test_to_dict_minimal(self):
        field = DenseVectorField(dimension=1536, metric="cosine")
        result = field.to_dict()
        assert result == {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}

    def test_to_dict_with_euclidean(self):
        field = DenseVectorField(dimension=768, metric="euclidean")
        result = field.to_dict()
        assert result == {"type": "dense_vector", "dimension": 768, "metric": "euclidean"}

    def test_to_dict_with_dotproduct(self):
        field = DenseVectorField(dimension=384, metric="dotproduct")
        result = field.to_dict()
        assert result == {"type": "dense_vector", "dimension": 384, "metric": "dotproduct"}

    def test_to_dict_with_description(self):
        field = DenseVectorField(dimension=1536, metric="cosine", description="OpenAI embeddings")
        result = field.to_dict()
        assert result == {
            "type": "dense_vector",
            "dimension": 1536,
            "metric": "cosine",
            "description": "OpenAI embeddings",
        }


class TestSparseVectorField:
    def test_default_values(self):
        field = SparseVectorField()
        assert field.metric == "dotproduct"
        assert field.description is None

    def test_to_dict_minimal(self):
        field = SparseVectorField()
        result = field.to_dict()
        assert result == {"type": "sparse_vector", "metric": "dotproduct"}

    def test_to_dict_with_description(self):
        field = SparseVectorField(description="BM25 sparse vectors")
        result = field.to_dict()
        assert result == {
            "type": "sparse_vector",
            "metric": "dotproduct",
            "description": "BM25 sparse vectors",
        }


class TestSemanticTextField:
    def test_required_params(self):
        field = SemanticTextField(model="multilingual-e5-large", field_map={"text": "content"})
        assert field.model == "multilingual-e5-large"
        assert field.field_map == {"text": "content"}
        assert field.read_parameters is None
        assert field.write_parameters is None
        assert field.description is None

    def test_to_dict_minimal(self):
        field = SemanticTextField(model="multilingual-e5-large", field_map={"text": "content"})
        result = field.to_dict()
        assert result == {
            "type": "semantic_text",
            "model": "multilingual-e5-large",
            "field_map": {"text": "content"},
        }

    def test_to_dict_with_parameters(self):
        field = SemanticTextField(
            model="multilingual-e5-large",
            field_map={"text": "content"},
            read_parameters={"truncate": "END"},
            write_parameters={"truncate": "START"},
        )
        result = field.to_dict()
        assert result == {
            "type": "semantic_text",
            "model": "multilingual-e5-large",
            "field_map": {"text": "content"},
            "read_parameters": {"truncate": "END"},
            "write_parameters": {"truncate": "START"},
        }

    def test_to_dict_with_description(self):
        field = SemanticTextField(
            model="multilingual-e5-large",
            field_map={"text": "content"},
            description="Semantic search field",
        )
        result = field.to_dict()
        assert result == {
            "type": "semantic_text",
            "model": "multilingual-e5-large",
            "field_map": {"text": "content"},
            "description": "Semantic search field",
        }
