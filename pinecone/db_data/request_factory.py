from __future__ import annotations

import logging
from typing import Any

from pinecone.core.openapi.db_data.models import (
    QueryRequest,
    UpsertRequest,
    DeleteRequest,
    UpdateRequest,
    DescribeIndexStatsRequest,
    FetchByMetadataRequest,
    SearchRecordsRequest,
    SearchRecordsRequestQuery,
    SearchRecordsRequestRerank,
    SearchMatchTerms,
    VectorValues,
    SearchRecordsVector,
    UpsertRecord,
    Vector as OpenApiVector,
)
from ..utils import parse_non_empty_args, convert_enum_to_string
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
    SearchQueryTypedDict,
    SearchRerankTypedDict,
    SearchQueryVectorTypedDict,
)

from .dataclasses import Vector, SparseValues, SearchQuery, SearchRerank, SearchQueryVector

logger = logging.getLogger(__name__)
""" :meta private: """


def non_openapi_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k not in OPENAPI_ENDPOINT_PARAMS}


class IndexRequestFactory:
    @staticmethod
    def query_request(
        top_k: int,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: SparseValues | SparseVectorTypedDict | None = None,
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

        result: QueryRequest = QueryRequest(
            **args_dict, _check_type=kwargs.pop("_check_type", False), **non_openapi_kwargs(kwargs)
        )
        return result

    @staticmethod
    def upsert_request(
        vectors: (
            list[Vector] | list[VectorTuple] | list[VectorTupleWithMetadata] | list[VectorTypedDict]
        ),
        namespace: str | None,
        _check_type: bool,
        **kwargs,
    ) -> UpsertRequest:
        args_dict = parse_non_empty_args([("namespace", namespace)])

        def vec_builder(
            v: Vector | VectorTuple | VectorTupleWithMetadata | VectorTypedDict,
        ) -> OpenApiVector:
            return VectorFactory.build(v, check_type=_check_type)

        result: UpsertRequest = UpsertRequest(
            vectors=list(map(vec_builder, vectors)),
            **args_dict,
            _check_type=_check_type,
            **non_openapi_kwargs(kwargs),
        )
        return result

    @staticmethod
    def delete_request(
        ids: list[str] | None = None,
        delete_all: bool | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        **kwargs,
    ) -> DeleteRequest:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args(
            [("ids", ids), ("delete_all", delete_all), ("namespace", namespace), ("filter", filter)]
        )
        result: DeleteRequest = DeleteRequest(
            **args_dict, **non_openapi_kwargs(kwargs), _check_type=_check_type
        )
        return result

    @staticmethod
    def fetch_by_metadata_request(
        filter: FilterTypedDict,
        namespace: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        **kwargs,
    ) -> FetchByMetadataRequest:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args(
            [
                ("namespace", namespace),
                ("filter", filter),
                ("limit", limit),
                ("pagination_token", pagination_token),
            ]
        )
        result: FetchByMetadataRequest = FetchByMetadataRequest(
            **args_dict, **non_openapi_kwargs(kwargs), _check_type=_check_type
        )
        return result

    @staticmethod
    def update_request(
        id: str | None = None,
        values: list[float] | None = None,
        set_metadata: VectorMetadataTypedDict | None = None,
        namespace: str | None = None,
        sparse_values: SparseValues | SparseVectorTypedDict | None = None,
        filter: FilterTypedDict | None = None,
        dry_run: bool | None = None,
        **kwargs,
    ) -> UpdateRequest:
        _check_type = kwargs.pop("_check_type", False)
        sparse_values_normalized = SparseValuesFactory.build(sparse_values)
        args_dict = parse_non_empty_args(
            [
                ("id", id),
                ("values", values),
                ("set_metadata", set_metadata),
                ("namespace", namespace),
                ("sparse_values", sparse_values_normalized),
                ("filter", filter),
                ("dry_run", dry_run),
            ]
        )

        result: UpdateRequest = UpdateRequest(
            **args_dict, _check_type=_check_type, **non_openapi_kwargs(kwargs)
        )
        return result

    @staticmethod
    def describe_index_stats_request(
        filter: FilterTypedDict | None = None, **kwargs
    ) -> DescribeIndexStatsRequest:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args([("filter", filter)])

        result: DescribeIndexStatsRequest = DescribeIndexStatsRequest(
            **args_dict, **non_openapi_kwargs(kwargs), _check_type=_check_type
        )
        return result

    @staticmethod
    def list_paginated_args(
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return parse_non_empty_args(
            [
                ("prefix", prefix),
                ("limit", limit),
                ("namespace", namespace),
                ("pagination_token", pagination_token),
            ]
        )

    @staticmethod
    def search_request(
        query: SearchQueryTypedDict | SearchQuery,
        rerank: SearchRerankTypedDict | SearchRerank | None = None,
        fields: list[str] | None = ["*"],  # Default to returning all fields
    ) -> SearchRecordsRequest:
        request_args = parse_non_empty_args(
            [
                ("query", IndexRequestFactory._parse_search_query(query)),
                ("fields", fields),
                ("rerank", IndexRequestFactory._parse_search_rerank(rerank)),
            ]
        )

        result: SearchRecordsRequest = SearchRecordsRequest(**request_args)
        return result

    @staticmethod
    def _parse_search_query(query: SearchQueryTypedDict | SearchQuery) -> SearchRecordsRequestQuery:
        if isinstance(query, SearchQuery):
            query_dict = query.as_dict()
        else:
            # query is SearchQueryTypedDict which is a TypedDict, so it's already a dict
            query_dict = query  # type: ignore[assignment]

        required_fields = {"top_k"}
        for key in required_fields:
            if query_dict.get(key, None) is None:
                raise ValueError(f"Missing required field '{key}' in search query.")

        # User-provided dict could contain object that need conversion
        if isinstance(query_dict.get("vector", None), SearchQueryVector):
            query_dict["vector"] = query_dict["vector"].as_dict()

        # Extract match_terms for conversion if present
        match_terms = query_dict.pop("match_terms", None)
        if match_terms is not None and isinstance(match_terms, dict):
            match_terms = SearchMatchTerms(**match_terms)

        srrq = SearchRecordsRequestQuery(
            **{k: v for k, v in query_dict.items() if k not in {"vector"}}
        )
        if query_dict.get("vector", None) is not None:
            srrq.vector = IndexRequestFactory._parse_search_vector(query_dict["vector"])
        if match_terms is not None:
            srrq.match_terms = match_terms
        result: SearchRecordsRequestQuery = srrq
        return result

    @staticmethod
    def _parse_search_vector(
        vector: SearchQueryVectorTypedDict | SearchQueryVector | None,
    ) -> SearchRecordsVector | None:
        if vector is None:
            return None

        if isinstance(vector, SearchQueryVector):
            if vector.values is None and vector.sparse_values is None:
                return None
            vector_dict = vector.as_dict()
        else:
            # vector is SearchQueryVectorTypedDict which is a TypedDict, so it's already a dict
            vector_dict = vector  # type: ignore[assignment]
            if (
                vector_dict.get("values", None) is None
                and vector_dict.get("sparse_values", None) is None
            ):
                return None

        from typing import cast

        srv = SearchRecordsVector(**{k: v for k, v in vector_dict.items() if k not in {"values"}})

        values = vector_dict.get("values", None)
        if values is not None:
            srv.values = VectorValues(value=values)

        return cast(SearchRecordsVector, srv)

    @staticmethod
    def _parse_search_rerank(
        rerank: SearchRerankTypedDict | SearchRerank | None = None,
    ) -> SearchRecordsRequestRerank | None:
        if rerank is None:
            return None

        if isinstance(rerank, SearchRerank):
            rerank_dict = rerank.as_dict()
        else:
            # rerank is SearchRerankTypedDict which is a TypedDict, so it's already a dict
            rerank_dict = rerank  # type: ignore[assignment]

        required_fields = {"model", "rank_fields"}
        for key in required_fields:
            if rerank_dict.get(key, None) is None:
                raise ValueError(f"Missing required field '{key}' in rerank.")

        rerank_dict["model"] = convert_enum_to_string(rerank_dict["model"])

        result: SearchRecordsRequestRerank = SearchRecordsRequestRerank(**rerank_dict)
        return result

    @staticmethod
    def upsert_records_args(namespace: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        if namespace is None:
            raise ValueError("namespace is required when upserting records")
        if not records or len(records) == 0:
            raise ValueError("No records provided")

        records_to_upsert = []
        for record in records:
            if not record.get("_id") and not record.get("id"):
                raise ValueError("Each record must have an '_id' or 'id' value")

            records_to_upsert.append(
                UpsertRecord(
                    record.get("_id", record.get("id")),
                    **{k: v for k, v in record.items() if k not in {"_id", "id"}},
                )
            )

        return {"namespace": namespace, "upsert_record": records_to_upsert}
