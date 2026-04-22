"""Tests for bulk import parquet validation.

Validation logic is adapted from the internal notebook
``bulk_import_parquet_validate.ipynb``.
"""

import json
import pytest

pytest.importorskip("pyarrow", reason="pyarrow required for bulk import validation")

import pyarrow as pa
import pyarrow.parquet as pq

from pinecone.db_data.resources.sync.bulk_import_validator import (
    validate_bulk_import_uri,
    _validate_schema,
    _validate_data_sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_schema(fields: dict) -> pa.Schema:
    """Build a pyarrow Schema from a {name: type} dict."""
    return pa.schema([pa.field(name, dtype) for name, dtype in fields.items()])


def make_dense_schema(dimension: int = 4, float_type=pa.float32()) -> pa.Schema:
    return make_schema(
        {
            "id": pa.string(),
            "values": pa.list_(float_type),
        }
    )


def make_fixed_dense_schema(dimension: int = 4) -> pa.Schema:
    return make_schema(
        {
            "id": pa.string(),
            "values": pa.list_(pa.field("item", pa.float32()), dimension),
        }
    )


def make_sparse_struct_type() -> pa.StructType:
    """The exact struct type Pinecone uses for sparse vectors in parquet."""
    return pa.struct([
        pa.field("indices", pa.list_(pa.uint32())),
        pa.field("values", pa.list_(pa.float32())),
    ])


def make_sparse_schema() -> pa.Schema:
    return make_schema(
        {
            "id": pa.string(),
            "sparse_values": make_sparse_struct_type(),
        }
    )


def make_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    arrays = {}
    for field in schema:
        arrays[field.name] = pa.array(
            [r.get(field.name) for r in rows], type=field.type
        )
    return pa.table(arrays, schema=schema)


def make_dense_table(
    n: int = 3,
    dimension: int = 4,
    float_type=pa.float32(),
    bad_id: bool = False,
    null_id: bool = False,
    null_vector: bool = False,
    non_finite: bool = False,
) -> pa.Table:
    ids = [None if (null_id and i == 0) else ("" if (bad_id and i == 0) else f"vec-{i}") for i in range(n)]
    vectors = []
    for i in range(n):
        if null_vector and i == 0:
            vectors.append(None)
        elif non_finite and i == 0:
            vectors.append([float("inf")] + [float(j) for j in range(dimension - 1)])
        else:
            vectors.append([float(j) for j in range(dimension)])
    return pa.table(
        {"id": pa.array(ids, pa.string()), "values": pa.array(vectors, pa.list_(float_type))},
    )


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_dense_schema(self):
        errors, warnings = [], []
        _validate_schema(make_dense_schema(), None, None, errors, warnings)
        assert errors == []

    def test_missing_id(self):
        schema = make_schema({"values": pa.list_(pa.float32())})
        errors, warnings = [], []
        _validate_schema(schema, None, None, errors, warnings)
        assert any("'id'" in e for e in errors)

    def test_id_wrong_type(self):
        schema = make_schema({"id": pa.int64(), "values": pa.list_(pa.float32())})
        errors, warnings = [], []
        _validate_schema(schema, None, None, errors, warnings)
        assert any("'id'" in e and "string" in e for e in errors)

    def test_missing_values_for_dense(self):
        schema = make_schema({"id": pa.string()})
        errors, warnings = [], []
        _validate_schema(schema, None, "dense", errors, warnings)
        assert any("'values'" in e for e in errors)

    def test_values_wrong_type(self):
        schema = make_schema({"id": pa.string(), "values": pa.list_(pa.string())})
        errors, warnings = [], []
        _validate_schema(schema, None, "dense", errors, warnings)
        assert any("'values'" in e and "float" in e for e in errors)

    def test_values_float64_accepted(self):
        schema = make_schema({"id": pa.string(), "values": pa.list_(pa.float64())})
        errors, warnings = [], []
        _validate_schema(schema, None, "dense", errors, warnings)
        assert errors == []

    def test_fixed_size_list_dimension_match(self):
        schema = make_fixed_dense_schema(dimension=4)
        errors, warnings = [], []
        _validate_schema(schema, 4, "dense", errors, warnings)
        assert errors == []

    def test_fixed_size_list_dimension_mismatch(self):
        schema = make_fixed_dense_schema(dimension=4)
        errors, warnings = [], []
        _validate_schema(schema, 8, "dense", errors, warnings)
        assert any("dimension" in e for e in errors)

    def test_fixed_size_list_dimension_inferred_in_warning(self):
        schema = make_fixed_dense_schema(dimension=4)
        errors, warnings = [], []
        _validate_schema(schema, None, "dense", errors, warnings)
        assert errors == []
        assert any("4" in w for w in warnings)

    def test_sparse_schema_valid(self):
        errors, warnings = [], []
        _validate_schema(make_sparse_schema(), None, "sparse", errors, warnings)
        assert errors == []

    def test_sparse_missing_sparse_values(self):
        schema = make_schema({"id": pa.string()})
        errors, warnings = [], []
        _validate_schema(schema, None, "sparse", errors, warnings)
        assert any("sparse_values" in e for e in errors)

    def test_sparse_wrong_type_flat_list(self):
        # Old (incorrect) two-column format should be caught
        schema = make_schema(
            {
                "id": pa.string(),
                "sparse_values": pa.list_(pa.float32()),
            }
        )
        errors, warnings = [], []
        _validate_schema(schema, None, "sparse", errors, warnings)
        assert any("STRUCT" in e for e in errors)

    def test_sparse_struct_missing_indices_field(self):
        bad_struct = pa.struct([pa.field("values", pa.list_(pa.float32()))])
        schema = make_schema({"id": pa.string(), "sparse_values": bad_struct})
        errors, warnings = [], []
        _validate_schema(schema, None, "sparse", errors, warnings)
        assert any("STRUCT" in e for e in errors)

    def test_sparse_struct_wrong_indices_type(self):
        bad_struct = pa.struct([
            pa.field("indices", pa.list_(pa.float32())),   # should be integer
            pa.field("values", pa.list_(pa.float32())),
        ])
        schema = make_schema({"id": pa.string(), "sparse_values": bad_struct})
        errors, warnings = [], []
        _validate_schema(schema, None, "sparse", errors, warnings)
        assert any("STRUCT" in e for e in errors)

    def test_no_vector_columns_detected(self):
        schema = make_schema({"id": pa.string(), "category": pa.string()})
        errors, warnings = [], []
        _validate_schema(schema, None, None, errors, warnings)
        assert any("No vector columns" in e for e in errors)

    def test_extra_columns_are_error(self):
        # Docs: "No additional columns permitted"
        schema = make_schema(
            {"id": pa.string(), "values": pa.list_(pa.float32()), "source": pa.string()}
        )
        errors, warnings = [], []
        _validate_schema(schema, None, "dense", errors, warnings)
        assert any("source" in e for e in errors)

    def test_metadata_string_column_ok(self):
        schema = make_schema(
            {
                "id": pa.string(),
                "values": pa.list_(pa.float32()),
                "metadata": pa.string(),
            }
        )
        errors, warnings = [], []
        _validate_schema(schema, None, "dense", errors, warnings)
        assert errors == []

    def test_metadata_struct_is_error(self):
        # Docs only document JSON-encoded UTF-8 string for metadata
        schema = make_schema(
            {
                "id": pa.string(),
                "values": pa.list_(pa.float32()),
                "metadata": pa.struct([pa.field("genre", pa.string())]),
            }
        )
        errors, warnings = [], []
        _validate_schema(schema, None, "dense", errors, warnings)
        assert any("metadata" in e for e in errors)

    def test_metadata_wrong_type_is_error(self):
        schema = make_schema(
            {
                "id": pa.string(),
                "values": pa.list_(pa.float32()),
                "metadata": pa.int64(),
            }
        )
        errors, warnings = [], []
        _validate_schema(schema, None, "dense", errors, warnings)
        assert any("metadata" in e for e in errors)


# ---------------------------------------------------------------------------
# Data sample validation tests
# ---------------------------------------------------------------------------


class TestValidateDataSample:
    def test_valid_dense_rows(self):
        table = make_dense_table(n=5, dimension=4)
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert errors == []

    def test_null_id(self):
        table = make_dense_table(n=3, null_id=True)
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert any("null" in e.lower() and "ID" in e for e in errors)

    def test_empty_id(self):
        table = make_dense_table(n=3, bad_id=True)
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert any("empty" in e.lower() for e in errors)

    def test_null_vector(self):
        table = make_dense_table(n=3, null_vector=True)
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert any("null" in e.lower() and "vector" in e.lower() for e in errors)

    def test_dimension_mismatch(self):
        table = make_dense_table(n=3, dimension=4)
        errors, warnings = [], []
        _validate_data_sample(table, 8, errors, warnings)
        assert any("dimension" in e.lower() or "length" in e.lower() for e in errors)

    def test_non_finite_value(self):
        table = make_dense_table(n=3, dimension=4, non_finite=True)
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert any("finite" in e.lower() or "Inf" in e for e in errors)

    def test_nan_value(self):
        table = pa.table(
            {
                "id": pa.array(["a", "b"], pa.string()),
                "values": pa.array(
                    [[float("nan"), 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]],
                    pa.list_(pa.float32()),
                ),
            }
        )
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert any("finite" in e.lower() or "NaN" in e for e in errors)

    def test_valid_metadata(self):
        meta = json.dumps({"genre": "fiction", "year": 2024, "tags": ["a", "b"]})
        table = pa.table(
            {
                "id": pa.array(["a"], pa.string()),
                "values": pa.array([[1.0, 2.0, 3.0, 4.0]], pa.list_(pa.float32())),
                "metadata": pa.array([meta], pa.string()),
            }
        )
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert errors == []

    def test_metadata_too_large(self):
        big_meta = json.dumps({"key": "x" * (41 * 1024)})
        table = pa.table(
            {
                "id": pa.array(["a"], pa.string()),
                "values": pa.array([[1.0, 2.0, 3.0, 4.0]], pa.list_(pa.float32())),
                "metadata": pa.array([big_meta], pa.string()),
            }
        )
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert any("40 KB" in e or "limit" in e for e in errors)

    def test_metadata_invalid_json(self):
        table = pa.table(
            {
                "id": pa.array(["a"], pa.string()),
                "values": pa.array([[1.0, 2.0, 3.0, 4.0]], pa.list_(pa.float32())),
                "metadata": pa.array(["not json"], pa.string()),
            }
        )
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert any("JSON" in e for e in errors)

    def test_metadata_not_dict(self):
        table = pa.table(
            {
                "id": pa.array(["a"], pa.string()),
                "values": pa.array([[1.0, 2.0, 3.0, 4.0]], pa.list_(pa.float32())),
                "metadata": pa.array([json.dumps([1, 2, 3])], pa.string()),
            }
        )
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert any("object" in e.lower() or "dict" in e.lower() for e in errors)

    def test_metadata_no_compatible_fields_warning(self):
        table = pa.table(
            {
                "id": pa.array(["a"], pa.string()),
                "values": pa.array([[1.0, 2.0, 3.0, 4.0]], pa.list_(pa.float32())),
                "metadata": pa.array([json.dumps({"key": [1, 2, 3]})], pa.string()),
            }
        )
        errors, warnings = [], []
        _validate_data_sample(table, 4, errors, warnings)
        assert errors == []
        assert any("compatible" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# End-to-end: validate_bulk_import_uri with real parquet files on disk
# ---------------------------------------------------------------------------


class TestValidateBulkImportUri:
    def test_valid_single_file(self, tmp_path):
        table = make_dense_table(n=10, dimension=4)
        path = str(tmp_path / "vectors.parquet")
        pq.write_table(table, path)

        result = validate_bulk_import_uri(path, dimension=4)
        assert result.is_valid
        assert result.files_checked == 1
        assert result.rows_sampled == 10

    def test_dimension_mismatch_end_to_end(self, tmp_path):
        table = make_dense_table(n=5, dimension=4)
        path = str(tmp_path / "vectors.parquet")
        pq.write_table(table, path)

        result = validate_bulk_import_uri(path, dimension=8)
        assert not result.is_valid
        assert any("dimension" in e.lower() or "length" in e.lower() for e in result.errors)

    def test_schema_only_no_rows_sampled(self, tmp_path):
        table = make_dense_table(n=10, dimension=4)
        path = str(tmp_path / "vectors.parquet")
        pq.write_table(table, path)

        result = validate_bulk_import_uri(path, dimension=4, sample_rows=0)
        assert result.is_valid
        assert result.rows_sampled == 0

    def test_directory_with_multiple_files(self, tmp_path):
        for i in range(3):
            table = make_dense_table(n=5, dimension=4)
            pq.write_table(table, str(tmp_path / f"part-{i}.parquet"))

        result = validate_bulk_import_uri(str(tmp_path), dimension=4)
        assert result.is_valid
        assert result.files_checked == 3

    def test_empty_directory(self, tmp_path):
        result = validate_bulk_import_uri(str(tmp_path))
        assert not result.is_valid
        assert any("No parquet files" in e for e in result.errors)

    def test_missing_id_column_end_to_end(self, tmp_path):
        table = pa.table({"values": pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float32()))})
        path = str(tmp_path / "bad.parquet")
        pq.write_table(table, path)

        result = validate_bulk_import_uri(path)
        assert not result.is_valid
        assert any("'id'" in e for e in result.errors)

    def test_result_repr_invalid(self, tmp_path):
        table = pa.table({"values": pa.array([[1.0, 2.0]], pa.list_(pa.float32()))})
        path = str(tmp_path / "bad.parquet")
        pq.write_table(table, path)

        result = validate_bulk_import_uri(path)
        r = repr(result)
        assert "INVALID" in r
        assert "'id'" in r

    def test_result_repr_valid(self, tmp_path):
        table = make_dense_table(n=2, dimension=4)
        path = str(tmp_path / "ok.parquet")
        pq.write_table(table, path)

        result = validate_bulk_import_uri(path, dimension=4)
        assert "VALID" in repr(result)

    def test_pyarrow_not_installed(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("pyarrow"):
                raise ImportError("No module named 'pyarrow'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="pinecone\\[parquet\\]"):
            validate_bulk_import_uri("/some/path.parquet")
