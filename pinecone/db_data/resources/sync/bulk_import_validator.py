from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from pinecone.db_data.dataclasses.bulk_import_validation_result import (
    BulkImportValidationResult,
)

if TYPE_CHECKING:
    import pyarrow as pa
    import pyarrow.parquet as pq

# Matches Pinecone's documented metadata size limit.
_MAX_METADATA_BYTES = 40 * 1024

# Scalar types Pinecone accepts as metadata values.
_VALID_METADATA_SCALAR_TYPES = (str, int, float, bool)

# Columns that have special meaning in bulk import parquet files.
# Note: sparse indices/values are sub-fields of the 'sparse_values' struct, not top-level columns.
_KNOWN_COLUMNS = {"id", "values", "sparse_values", "metadata"}


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyarrow is required for bulk import validation. "
            "Install it with: pip install 'pinecone[parquet]'"
        )


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _is_string_type(t) -> bool:
    import pyarrow as pa

    return bool(pa.types.is_string(t) or pa.types.is_large_string(t))


def _is_float_list_type(t) -> bool:
    import pyarrow as pa

    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return bool(pa.types.is_floating(t.value_type))
    return False


def _is_integer_list_type(t) -> bool:
    import pyarrow as pa

    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return bool(pa.types.is_integer(t.value_type))
    return False


def _fixed_list_size(t) -> int | None:
    """Return the list size if the type is fixed_size_list, else None."""
    import pyarrow as pa

    if pa.types.is_fixed_size_list(t):
        return int(t.list_size)
    return None


def _is_sparse_struct_type(t) -> bool:
    """Check if t matches STRUCT<indices: LIST<uint>, values: LIST<float>>.

    Confirmed format from real Pinecone parquet files:
      sparse_values: struct<indices: list<element: uint32>, values: list<element: float>>
    """
    import pyarrow as pa

    if not pa.types.is_struct(t):
        return False
    field_names = {t.field(i).name for i in range(t.num_fields)}
    if "indices" not in field_names or "values" not in field_names:
        return False
    return bool(
        _is_integer_list_type(t.field("indices").type)
        and _is_float_list_type(t.field("values").type)
    )


# ---------------------------------------------------------------------------
# Schema validation (reads only the parquet footer — no data download)
# ---------------------------------------------------------------------------


def _validate_schema(
    schema: "pa.Schema",
    dimension: int | None,
    vector_type: str | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    field_names = set(schema.names)

    if "id" not in field_names:
        errors.append("Missing required column 'id'")
    else:
        t = schema.field("id").type
        if not _is_string_type(t):
            errors.append(f"Column 'id' must be string type, got {t}")

    has_values = "values" in field_names
    has_sparse = "sparse_values" in field_names

    is_dense = (vector_type == "dense") if vector_type else has_values
    is_sparse = (vector_type == "sparse") if vector_type else has_sparse

    if not is_dense and not is_sparse:
        errors.append(
            "No vector columns detected. "
            "Expected a 'values' column (dense) or a 'sparse_values' struct column (sparse)."
        )
        return

    if is_dense:
        if "values" not in field_names:
            errors.append("Missing required column 'values' for dense vectors")
        else:
            t = schema.field("values").type
            if not _is_float_list_type(t):
                errors.append(f"Column 'values' must be a list of floats, got {t}")
            else:
                schema_dim = _fixed_list_size(t)
                if schema_dim is not None and dimension is not None and schema_dim != dimension:
                    errors.append(
                        f"Vector dimension in schema ({schema_dim}) does not match "
                        f"expected dimension ({dimension})"
                    )
                elif schema_dim is not None and dimension is None:
                    warnings.append(f"Detected vector dimension from schema: {schema_dim}")

    if is_sparse:
        if "sparse_values" not in field_names:
            errors.append("Missing required column 'sparse_values' for sparse vectors")
        else:
            t = schema.field("sparse_values").type
            if not _is_sparse_struct_type(t):
                errors.append(
                    f"Column 'sparse_values' must be "
                    f"STRUCT<indices: LIST<uint32>, values: LIST<float>>, got {t}"
                )

    if "metadata" in field_names:
        t = schema.field("metadata").type
        if not _is_string_type(t):
            errors.append(
                f"Column 'metadata' must be a JSON-encoded UTF-8 string, got {t}. "
                "See https://docs.pinecone.io/guides/index-data/import-data"
            )

    extra = field_names - _KNOWN_COLUMNS
    if extra:
        errors.append(
            f"Unexpected column(s) {sorted(extra)} — no additional columns are permitted. "
            "Only 'id', 'values', 'sparse_values', and 'metadata' are allowed."
        )


# ---------------------------------------------------------------------------
# Data validation (reads a small sample of rows)
# ---------------------------------------------------------------------------


def _is_valid_metadata_value(v) -> bool:
    if isinstance(v, _VALID_METADATA_SCALAR_TYPES):
        return True
    if isinstance(v, list):
        return all(isinstance(x, str) for x in v)
    return False


def _validate_data_sample(
    table: "pa.Table",
    dimension: int | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    import pyarrow as pa

    if "id" in table.schema.names:
        id_col = table.column("id")
        if id_col.null_count > 0:
            errors.append(f"Found {id_col.null_count} null ID(s)")
        empty = sum(1 for v in id_col if v.is_valid and v.as_py() == "")
        if empty:
            errors.append(f"Found {empty} empty string ID(s)")

    if "values" in table.schema.names:
        values_col = table.column("values")
        if values_col.null_count > 0:
            errors.append(f"Found {values_col.null_count} null vector(s) in 'values'")

        for i, val in enumerate(values_col):
            if not val.is_valid:
                continue
            arr = val.as_py()
            if arr is None:
                continue
            if dimension is not None and len(arr) != dimension:
                errors.append(
                    f"Row {i}: vector length {len(arr)} != expected dimension {dimension}"
                )
                break
            non_finite = [x for x in arr if x is None or not math.isfinite(x)]
            if non_finite:
                errors.append(f"Row {i}: 'values' contains non-finite value(s) (NaN or Inf)")
                break

    if "metadata" in table.schema.names:
        meta_col = table.column("metadata")
        # Only validate JSON-string metadata; struct columns are validated by the schema check.
        if pa.types.is_string(meta_col.type) or pa.types.is_large_string(meta_col.type):
            for i, val in enumerate(meta_col):
                if not val.is_valid:
                    continue
                raw = val.as_py()
                if raw is None:
                    continue
                size = len(raw.encode("utf-8"))
                if size > _MAX_METADATA_BYTES:
                    errors.append(
                        f"Row {i}: metadata size {size} bytes exceeds the 40 KB limit"
                    )
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    errors.append(f"Row {i}: metadata is not valid JSON: {e}")
                    continue
                if not isinstance(obj, dict):
                    errors.append(
                        f"Row {i}: metadata must be a JSON object, got {type(obj).__name__}"
                    )
                    continue
                valid_fields = {k: v for k, v in obj.items() if _is_valid_metadata_value(v)}
                if obj and not valid_fields:
                    warnings.append(
                        f"Row {i}: metadata has no Pinecone-compatible fields "
                        "(values must be string, number, bool, or list of strings)"
                    )


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------


def _list_parquet_files(uri: str) -> list[str]:
    """Return all parquet file URIs under a path (handles single file or directory)."""
    import pyarrow.fs as pafs

    if uri.lower().endswith(".parquet"):
        return [uri]

    fs, root_path = pafs.FileSystem.from_uri(uri)
    scheme = (uri.split("://")[0] + "://") if "://" in uri else ""

    root_path = root_path.rstrip("/")
    selector = pafs.FileSelector(root_path, recursive=True)
    file_infos = fs.get_file_info(selector)

    result = []
    for fi in file_infos:
        if fi.type == pafs.FileType.File and fi.base_name.lower().endswith(".parquet"):
            result.append(f"{scheme}{fi.path}" if scheme else fi.path)

    return sorted(result)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_bulk_import_uri(
    uri: str,
    dimension: int | None = None,
    vector_type: str | None = None,
    sample_rows: int = 100,
    verbose: bool = False,
) -> BulkImportValidationResult:
    """Validate parquet file(s) at *uri* for Pinecone bulk import compatibility.

    Schema validation reads only the parquet file footer — no vector data is
    downloaded — making it cheap even for large remote files. When
    ``sample_rows > 0`` a small number of rows are also read to check for null
    IDs, non-finite values, and metadata correctness.

    Args:
        uri: Local path or remote URI (``s3://``, ``gs://``, ``az://``).
            May point to a single ``.parquet`` file or a directory/prefix
            containing multiple files.
        dimension: Expected vector dimension. When provided, any mismatch
            between the file and this value is reported as an error.
        vector_type: ``"dense"`` or ``"sparse"``. Inferred from the schema
            when omitted.
        sample_rows: Number of rows to read for data-level checks. Set to
            ``0`` to perform schema-only validation without reading any data.
        verbose: When ``True``, print per-file progress and a summary to stdout.

    Returns:
        :class:`BulkImportValidationResult` — pass ``result.uri`` directly to
        ``index.bulk_import.start()`` if ``result.is_valid`` is ``True``.
    """
    _require_pyarrow()
    import pyarrow.parquet as pq

    errors: list[str] = []
    warnings: list[str] = []
    files_checked = 0
    rows_sampled = 0

    try:
        parquet_files = _list_parquet_files(uri)
    except Exception as e:
        return BulkImportValidationResult(
            is_valid=False,
            uri=uri,
            errors=[f"Failed to access '{uri}': {e}"],
        )

    if not parquet_files:
        return BulkImportValidationResult(
            is_valid=False,
            uri=uri,
            errors=[f"No parquet files found at '{uri}'"],
        )

    total = len(parquet_files)
    multi = total > 1

    if verbose:
        print(f"Validating {total} file(s) at {uri} ...")

    ok_count = 0
    bad_count = 0

    for file_uri in parquet_files:
        file_errors: list[str] = []
        file_warnings: list[str] = []
        index = files_checked + 1

        try:
            schema = pq.read_schema(file_uri)
        except Exception as e:
            msg = f"failed to read parquet schema: {e}"
            errors.append(f"{file_uri}: {msg}")
            if verbose:
                print(f"[{index:>{len(str(total))}}/{total}] BAD  {file_uri}")
                print(f"         {msg}")
            files_checked += 1
            bad_count += 1
            continue

        _validate_schema(schema, dimension, vector_type, file_errors, file_warnings)

        if sample_rows > 0 and not file_errors:
            try:
                columns = [c for c in _KNOWN_COLUMNS if c in schema.names]
                pf = pq.ParquetFile(file_uri)
                sample_table = None
                for batch in pf.iter_batches(batch_size=sample_rows, columns=columns):
                    import pyarrow as pa

                    sample_table = pa.Table.from_batches([batch])
                    break
                if sample_table is not None:
                    _validate_data_sample(sample_table, dimension, file_errors, file_warnings)
                    rows_sampled += len(sample_table)
            except Exception as e:
                file_warnings.append(f"Could not read sample data: {e}")

        if verbose:
            status = "BAD " if file_errors else "OK  "
            print(f"[{index:>{len(str(total))}}/{total}] {status} {file_uri}")
            for fe in file_errors:
                print(f"         error:   {fe}")
            for fw in file_warnings:
                print(f"         warning: {fw}")

        prefix = f"{file_uri}: " if multi else ""
        errors.extend(f"{prefix}{e}" for e in file_errors)
        warnings.extend(f"{prefix}{w}" for w in file_warnings)
        files_checked += 1
        if file_errors:
            bad_count += 1
        else:
            ok_count += 1

    if verbose:
        print(f"\nTotal: {total}  OK: {ok_count}  BAD: {bad_count}  rows sampled: {rows_sampled}")

    return BulkImportValidationResult(
        is_valid=len(errors) == 0,
        uri=uri,
        errors=errors,
        warnings=warnings,
        files_checked=files_checked,
        rows_sampled=rows_sampled,
    )
