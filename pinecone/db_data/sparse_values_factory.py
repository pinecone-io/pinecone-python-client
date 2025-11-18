from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..utils import convert_to_list

from .errors import (
    SparseValuesTypeError,
    SparseValuesMissingKeysError,
    SparseValuesDictionaryExpectedError,
)

from .dataclasses import SparseValues
from .types import SparseVectorTypedDict
from pinecone.core.openapi.db_data.models import SparseValues as OpenApiSparseValues


class SparseValuesFactory:
    """SparseValuesFactory is used to convert various types of user input into SparseValues objects used in generated request code."""

    @staticmethod
    def build(
        input: (SparseValues | OpenApiSparseValues | SparseVectorTypedDict) | None,
    ) -> OpenApiSparseValues | None:
        if input is None:
            return input
        if isinstance(input, OpenApiSparseValues):
            result_input: OpenApiSparseValues = input
            return result_input
        if isinstance(input, SparseValues):
            result: OpenApiSparseValues = OpenApiSparseValues(
                indices=input.indices, values=input.values
            )
            return result
        if not isinstance(input, Mapping):
            raise SparseValuesDictionaryExpectedError(input)
        if not {"indices", "values"}.issubset(input):
            raise SparseValuesMissingKeysError(input)

        indices = SparseValuesFactory._convert_to_list(input.get("indices"), int)
        values = SparseValuesFactory._convert_to_list(input.get("values"), float)

        if len(indices) != len(values):
            raise ValueError("Sparse values indices and values must have the same length")

        try:
            result_dict: OpenApiSparseValues = OpenApiSparseValues(indices=indices, values=values)
            return result_dict
        except TypeError as e:
            raise SparseValuesTypeError() from e

    @staticmethod
    def _convert_to_list(input: Any, expected_type: type) -> list[Any]:
        try:
            converted = convert_to_list(input)
        except TypeError as e:
            raise SparseValuesTypeError() from e

        SparseValuesFactory._validate_list_items_type(converted, expected_type)
        return converted  # type: ignore[no-any-return]

    @staticmethod
    def _validate_list_items_type(input: list[Any], expected_type: type) -> None:
        if len(input) > 0 and not isinstance(input[0], expected_type):
            raise SparseValuesTypeError()
