import logging
from typing import Dict, Any, Optional

from pinecone.core.openapi.repository_data.models import (
    QueryModel,
    SearchDocuments,
    QueryInputModel,
)

from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS


logger = logging.getLogger(__name__)
""" :meta private: """


def non_openapi_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k not in OPENAPI_ENDPOINT_PARAMS}


class RepositoryRequestFactory:
    @staticmethod
    def search_request(
        query_str: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        # TODO, add options
    ) -> SearchDocuments:
        filter = {} if filter is None else filter
        return SearchDocuments(
            query=QueryModel(inputs=QueryInputModel(text=query_str), top_k=top_k, filter=filter)
        )
