from collections.abc import Mapping
from typing import Union, Dict

from ..utils import convert_to_list

from ..data import (
    SparseValuesTypeError,
    SparseValuesMissingKeysError,
    SparseValuesDictionaryExpectedError,
)

from pinecone.core.grpc.protos.vector_service_pb2 import SparseValues as GRPCSparseValues
from pinecone import SparseValues as NonGRPCSparseValues


class SparseValuesFactory:
    @staticmethod
    def build(input: Union[Dict, GRPCSparseValues, NonGRPCSparseValues]) -> GRPCSparseValues:
        if input is None:
            return input
        if isinstance(input, GRPCSparseValues):
            return input
        if isinstance(input, NonGRPCSparseValues):
            return GRPCSparseValues(indices=input.indices, values=input.values)
        if not isinstance(input, Mapping):
            raise SparseValuesDictionaryExpectedError(input)
        if not {"indices", "values"}.issubset(input):
            raise SparseValuesMissingKeysError(input)

        indices = SparseValuesFactory._convert_to_list(input.get("indices"), int)
        values = SparseValuesFactory._convert_to_list(input.get("values"), float)

        if len(indices) != len(values):
            raise ValueError("Sparse values indices and values must have the same length")

        try:
            return GRPCSparseValues(indices=indices, values=values)
        except TypeError as e:
            raise SparseValuesTypeError() from e

    @staticmethod
    def _convert_to_list(input, expected_type):
        try:
            converted = convert_to_list(input)
        except TypeError as e:
            raise SparseValuesTypeError() from e

        SparseValuesFactory._validate_list_items_type(converted, expected_type)
        return converted

    @staticmethod
    def _validate_list_items_type(input, expected_type):
        if len(input) > 0 and not isinstance(input[0], expected_type):
            raise SparseValuesTypeError()
