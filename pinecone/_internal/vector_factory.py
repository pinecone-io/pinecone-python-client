"""Vector input format parsing — normalizes user inputs to canonical Vector objects."""

from __future__ import annotations

from typing import Any

from pinecone.errors.exceptions import PineconeTypeError, PineconeValueError
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector

_RECOGNIZED_KEYS = {"id", "values", "sparse_values", "metadata"}


class VectorFactory:
    """Converts user-provided vector inputs into canonical ``Vector`` objects.

    Accepted formats:

    - ``Vector`` instance (passthrough)
    - ``tuple`` of 2 elements ``(id, values)`` or 3 elements ``(id, values, metadata)``
    - ``dict`` with keys drawn from ``{"id", "values", "sparse_values", "metadata"}``
    """

    @staticmethod
    def build(item: Any) -> Vector:
        """Convert a user-provided vector input to a ``Vector`` object."""
        if isinstance(item, Vector):
            if not item.values and item.sparse_values is None:
                raise PineconeValueError(
                    "Vector must have at least one of non-empty dense values or sparse values"
                )
            return item
        if isinstance(item, tuple):
            return VectorFactory._from_tuple(item)
        if isinstance(item, dict):
            return VectorFactory._from_dict(item)
        raise PineconeTypeError(f"Expected Vector, tuple, or dict, got {type(item).__name__}")

    @staticmethod
    def _from_tuple(item: tuple[Any, ...]) -> Vector:
        length = len(item)
        if length == 2:
            id_, values = item
            VectorFactory._validate_id(id_)
            converted = list(values)
            if not converted:
                raise PineconeValueError(
                    "Vector must have at least one of non-empty dense values or sparse values"
                )
            return Vector(id=id_, values=converted)
        if length == 3:
            id_, values, metadata = item
            VectorFactory._validate_id(id_)
            if metadata is not None and not isinstance(metadata, dict):
                raise PineconeTypeError(f"metadata must be a dict, got {type(metadata).__name__}")
            converted = list(values)
            if not converted:
                raise PineconeValueError(
                    "Vector must have at least one of non-empty dense values or sparse values"
                )
            return Vector(id=id_, values=converted, metadata=metadata)
        raise PineconeValueError(f"Vector tuple must have 2 or 3 elements, got {length}")

    @staticmethod
    def _from_dict(item: dict[str, Any]) -> Vector:
        if "id" not in item:
            raise PineconeValueError("Vector dict must contain an 'id' key")
        extra = set(item.keys()) - _RECOGNIZED_KEYS
        if extra:
            raise PineconeValueError(f"Vector dict contains unrecognized keys: {sorted(extra)}")

        id_ = item["id"]
        VectorFactory._validate_id(id_)

        raw_values = item.get("values")
        values: list[float] = list(raw_values) if raw_values is not None else []

        raw_sparse = item.get("sparse_values")
        sparse: SparseValues | None = None
        if raw_sparse is not None:
            sparse = VectorFactory._parse_sparse(raw_sparse)

        metadata = item.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise PineconeTypeError(f"metadata must be a dict, got {type(metadata).__name__}")

        if not values and sparse is None:
            raise PineconeValueError(
                "Vector must have at least one of non-empty dense values or sparse values"
            )

        return Vector(
            id=id_,
            values=values,
            sparse_values=sparse,
            metadata=metadata,
        )

    @staticmethod
    def _parse_sparse(raw: Any) -> SparseValues:
        if not isinstance(raw, dict):
            raise PineconeTypeError(f"sparse_values must be a dict, got {type(raw).__name__}")
        if "indices" not in raw or "values" not in raw:
            missing = []
            if "indices" not in raw:
                missing.append("indices")
            if "values" not in raw:
                missing.append("values")
            raise PineconeValueError(f"sparse_values dict is missing required keys: {missing}")

        indices = raw["indices"]
        values = raw["values"]

        if len(indices) != len(values):
            raise PineconeValueError(
                f"sparse_values indices and values must have the same length, "
                f"got {len(indices)} and {len(values)}"
            )

        if indices:
            if not isinstance(indices[0], int):
                raise PineconeTypeError(
                    f"sparse_values indices must be integers, got {type(indices[0]).__name__}"
                )
        if values:
            if not isinstance(values[0], (int, float)):
                raise PineconeTypeError(
                    f"sparse_values values must be floats, got {type(values[0]).__name__}"
                )

        return SparseValues(
            indices=list(indices),
            values=[float(v) for v in values],
        )

    @staticmethod
    def _validate_id(id_: Any) -> None:
        if not isinstance(id_, str):
            raise PineconeTypeError(f"Vector ID must be a string, got {type(id_).__name__}")
        if not id_.isascii():
            raise PineconeValueError(f"Vector ID must contain only ASCII characters, got: {id_!r}")
        if "\x00" in id_:
            raise PineconeValueError(f"Vector ID must not contain null characters, got: {id_!r}")
