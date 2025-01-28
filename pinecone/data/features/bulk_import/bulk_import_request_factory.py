from enum import Enum
from typing import Optional, Literal, Type, TypedDict, cast, Any

from pinecone.core.openapi.db_data.models import (
    StartImportRequest,
    ImportErrorMode as ImportErrorModeClass,
)

from pinecone.utils import parse_non_empty_args

ImportErrorMode: Type[Enum] = cast(
    Type[Enum], Enum("ImportErrorMode", ImportErrorModeClass.allowed_values[("on_error",)])
)


class DescribeImportArgs(TypedDict, total=False):
    id: str


class CancelImportArgs(TypedDict, total=False):
    id: str


class BulkImportRequestFactory:
    @staticmethod
    def start_import_request(
        uri: str,
        integration_id: Optional[str] = None,
        error_mode: Optional[Literal["CONTINUE", "ABORT"]] = "CONTINUE",
    ) -> StartImportRequest:
        if isinstance(error_mode, ImportErrorMode):
            error_mode = error_mode.value
        elif isinstance(error_mode, str):
            try:
                error_mode = ImportErrorMode(error_mode.lower()).value
            except ValueError:
                raise ValueError(f"Invalid error_mode value: {error_mode}")

        args_dict = parse_non_empty_args(
            [
                ("uri", uri),
                ("integration_id", integration_id),
                ("error_mode", ImportErrorModeClass(on_error=error_mode)),
            ]
        )

        return StartImportRequest(**args_dict)

    @staticmethod
    def list_imports_paginated_args(
        limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> dict[str, Any]:
        return parse_non_empty_args([("limit", limit), ("pagination_token", pagination_token)])

    @staticmethod
    def describe_import_args(id: str) -> DescribeImportArgs:
        if isinstance(id, int):
            id = str(id)
        return {"id": id}

    @staticmethod
    def cancel_import_args(id: str) -> CancelImportArgs:
        if isinstance(id, int):
            id = str(id)
        return {"id": id}
