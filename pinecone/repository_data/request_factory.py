import logging
from typing import Union, Dict, Any, cast

from pinecone.core.openapi.repository_data.models import QueryModel, SearchDocuments

from pinecone.core.openapi.db_data.models import SearchRecordsRequest, SearchRecordsRequestQuery
from ..utils import parse_non_empty_args
from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS
from .types import SearchQueryTypedDict

from .dataclasses import SearchQuery

logger = logging.getLogger(__name__)
""" :meta private: """


def non_openapi_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k not in OPENAPI_ENDPOINT_PARAMS}


class RepositoryRequestFactory:
    @staticmethod
    def search_request(
        query: Union[SearchQueryTypedDict, SearchQuery],
        # TODO, add options
    ) -> SearchRecordsRequest:
        request_args = parse_non_empty_args(
            [
                ("query", RepositoryRequestFactory._parse_search_query(query))
                # TODO, add options
            ]
        )

        return SearchDocuments(**request_args)

    @staticmethod
    def _parse_search_query(
        query: Union[SearchQueryTypedDict, SearchQuery],
        # TODO, add options
    ) -> SearchRecordsRequestQuery:
        if isinstance(query, SearchQuery):
            query_dict = query.as_dict()
        else:
            query_dict = cast(dict[str, Any], query)

        if not query_dict.get("top_k"):
            query_dict["top_k"] = 10

        required_fields = {"inputs"}
        for key in required_fields:
            if query_dict.get(key, None) is None:
                raise ValueError(f"Missing required field '{key}' in search query.")

        query_model = QueryModel(**{k: v for k, v in query_dict.items()})
        return query_model
