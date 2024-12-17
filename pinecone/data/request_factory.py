import logging
from typing import Union, List, Optional, Dict, Any

from pinecone.core.openapi.db_data.models import (
    QueryRequest,
    UpsertRequest,
    DeleteRequest,
    UpdateRequest,
    DescribeIndexStatsRequest,
)
from ..utils import parse_non_empty_args
from .vector_factory import VectorFactory
from .sparse_values_factory import SparseValuesFactory
from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS
from .types import (
    VectorTypedDict,
    SparseVectorTypedDict,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
    FilterTypedDict,
)
from .dataclasses import Vector, SparseValues

logger = logging.getLogger(__name__)


def non_openapi_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if k not in OPENAPI_ENDPOINT_PARAMS}


class IndexRequestFactory:
    @staticmethod
    def query_request(
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> QueryRequest:
        if vector is not None and id is not None:
            raise ValueError("Cannot specify both `id` and `vector`")

        sparse_vector_normalized = SparseValuesFactory.build(sparse_vector)
        args_dict = parse_non_empty_args(
            [
                ("vector", vector),
                ("id", id),
                ("queries", None),
                ("top_k", top_k),
                ("namespace", namespace),
                ("filter", filter),
                ("include_values", include_values),
                ("include_metadata", include_metadata),
                ("sparse_vector", sparse_vector_normalized),
            ]
        )

        return QueryRequest(
            **args_dict, _check_type=kwargs.pop("_check_type", False), **non_openapi_kwargs(kwargs)
        )

    @staticmethod
    def upsert_request(
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
        ],
        namespace: Optional[str],
        _check_type: bool,
        **kwargs,
    ) -> UpsertRequest:
        args_dict = parse_non_empty_args([("namespace", namespace)])

        def vec_builder(v):
            return VectorFactory.build(v, check_type=_check_type)

        return UpsertRequest(
            vectors=list(map(vec_builder, vectors)),
            **args_dict,
            _check_type=_check_type,
            **non_openapi_kwargs(kwargs),
        )

    @staticmethod
    def delete_request(
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        **kwargs,
    ) -> DeleteRequest:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args(
            [("ids", ids), ("delete_all", delete_all), ("namespace", namespace), ("filter", filter)]
        )
        return DeleteRequest(**args_dict, **non_openapi_kwargs(kwargs), _check_type=_check_type)

    @staticmethod
    def update_request(
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> UpdateRequest:
        _check_type = kwargs.pop("_check_type", False)
        sparse_values_normalized = SparseValuesFactory.build(sparse_values)
        args_dict = parse_non_empty_args(
            [
                ("values", values),
                ("set_metadata", set_metadata),
                ("namespace", namespace),
                ("sparse_values", sparse_values_normalized),
            ]
        )

        return UpdateRequest(
            id=id, **args_dict, _check_type=_check_type, **non_openapi_kwargs(kwargs)
        )

    @staticmethod
    def describe_index_stats_request(
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None, **kwargs
    ) -> DescribeIndexStatsRequest:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args([("filter", filter)])

        return DescribeIndexStatsRequest(
            **args_dict, **non_openapi_kwargs(kwargs), _check_type=_check_type
        )

    @staticmethod
    def list_paginated_args(
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return parse_non_empty_args(
            [
                ("prefix", prefix),
                ("limit", limit),
                ("namespace", namespace),
                ("pagination_token", pagination_token),
            ]
        )
