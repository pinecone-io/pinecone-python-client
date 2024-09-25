import numbers

from collections.abc import Iterable, Mapping
from typing import Union, Tuple, Dict

from ..utils import fix_tuple_length, convert_to_list
from ..utils.constants import REQUIRED_VECTOR_FIELDS, OPTIONAL_VECTOR_FIELDS
from .sparse_vector_factory import SparseValuesFactory

from pinecone.core.openapi.data.models import Vector, SparseValues

from .errors import (
    VectorDictionaryMissingKeysError,
    VectorDictionaryExcessKeysError,
    VectorTupleLengthError,
    MetadataDictionaryExpectedError,
)


class VectorFactory:
    @staticmethod
    def build(item: Union[Vector, Tuple, Dict], check_type: bool = True) -> Vector:
        if isinstance(item, Vector):
            return item
        elif isinstance(item, tuple):
            return VectorFactory._tuple_to_vector(item, check_type)
        elif isinstance(item, Mapping):
            return VectorFactory._dict_to_vector(item, check_type)
        else:
            raise ValueError(f"Invalid vector value passed: cannot interpret type {type(item)}")

    @staticmethod
    def _tuple_to_vector(item, check_type: bool) -> Vector:
        if len(item) < 2 or len(item) > 3:
            raise VectorTupleLengthError(item)
        id, values, metadata = fix_tuple_length(item, 3)
        if isinstance(values, SparseValues):
            raise ValueError(
                "Sparse values are not supported in tuples. Please use either dicts or Vector objects as inputs."
            )
        else:
            return Vector(
                id=id,
                values=convert_to_list(values),
                metadata=metadata or {},
                _check_type=check_type,
            )

    @staticmethod
    def _dict_to_vector(item, check_type: bool) -> Vector:
        item_keys = set(item.keys())
        if not item_keys.issuperset(REQUIRED_VECTOR_FIELDS):
            raise VectorDictionaryMissingKeysError(item)

        excessive_keys = item_keys - (REQUIRED_VECTOR_FIELDS | OPTIONAL_VECTOR_FIELDS)
        if len(excessive_keys) > 0:
            raise VectorDictionaryExcessKeysError(item)

        values = item.get("values")
        if "values" in item:
            item["values"] = convert_to_list(values)

        sparse_values = item.get("sparse_values")
        if sparse_values is None:
            item.pop("sparse_values", None)
        else:
            item["sparse_values"] = SparseValuesFactory.build(sparse_values)

        metadata = item.get("metadata")
        if metadata and not isinstance(metadata, Mapping):
            raise MetadataDictionaryExpectedError(item)

        try:
            return Vector(**item, _check_type=check_type)
        except TypeError as e:
            if not isinstance(item["values"], Iterable) or not isinstance(
                item["values"].__iter__().__next__(), numbers.Real
            ):
                raise TypeError("Column `values` is expected to be a list of floats")
            raise e
