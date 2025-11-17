import logging

from pinecone.core.openapi.db_data.api.vector_operations_api import AsyncioVectorOperationsApi
from pinecone.core.openapi.db_data.models import SearchRecordsResponse
from pinecone.db_data.dataclasses import SearchQuery, SearchRerank, UpsertResponse
from pinecone.db_data.request_factory import IndexRequestFactory
from pinecone.db_data.types import SearchQueryTypedDict, SearchRerankTypedDict
from pinecone.utils import validate_and_convert_errors, PluginAware

logger = logging.getLogger(__name__)
""" :meta private: """


class RecordResourceAsyncio(PluginAware):
    """Resource for record operations on a Pinecone index (async)."""

    def __init__(self, vector_api: AsyncioVectorOperationsApi, config, openapi_config):
        self._vector_api = vector_api
        """ :meta private: """
        self._config = config
        """ :meta private: """
        self._openapi_config = openapi_config
        """ :meta private: """
        super().__init__()

    @validate_and_convert_errors
    async def upsert_records(self, namespace: str, records: list[dict]) -> UpsertResponse:
        """Upsert records to a namespace.

        A record is a dictionary that contains either an `id` or `_id` field along with
        other fields that will be stored as metadata. The `id` or `_id` field is used
        as the unique identifier for the record. At least one field in the record should
        correspond to a field mapping in the index's embed configuration.

        When records are upserted, Pinecone converts mapped fields into embeddings and
        upserts them into the specified namespace of the index.

        Args:
            namespace: The namespace of the index to upsert records to.
            records: The records to upsert into the index. Each record must have an 'id'
                or '_id' field.

        Returns:
            UpsertResponse object which contains the number of records upserted.

        Raises:
            ValueError: If namespace is not provided or if no records are provided, or
                if a record is missing an 'id' or '_id' field.

        Examples:
            >>> await index.record.upsert_records(
            ...     namespace='my-namespace',
            ...     records=[
            ...         {
            ...             "_id": "test1",
            ...             "my_text_field": "Apple is a popular fruit known for its sweetness.",
            ...         },
            ...         {
            ...             "_id": "test2",
            ...             "my_text_field": "The tech company Apple is known for its innovative products.",
            ...         },
            ...     ]
            ... )
        """
        args = IndexRequestFactory.upsert_records_args(namespace=namespace, records=records)
        # Use _return_http_data_only=False to get headers for LSN extraction
        result = await self._vector_api.upsert_records_namespace(
            _return_http_data_only=False, **args
        )
        # result is a tuple: (data, status, headers) when _return_http_data_only=False
        response_info = None
        if isinstance(result, tuple) and len(result) >= 3:
            headers = result[2]
            if headers:
                from pinecone.utils.response_info import extract_response_info

                response_info = extract_response_info(headers)
                # response_info may contain raw_headers even without LSN values

        # Ensure response_info is always present
        if response_info is None:
            from pinecone.utils.response_info import extract_response_info

            response_info = extract_response_info({})

        # Count records (could be len(records) but we don't know if any failed)
        # For now, assume all succeeded
        return UpsertResponse(upserted_count=len(records), _response_info=response_info)

    @validate_and_convert_errors
    async def search(
        self,
        namespace: str,
        query: SearchQueryTypedDict | SearchQuery,
        rerank: (SearchRerankTypedDict | SearchRerank) | None = None,
        fields: list[str] | None = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """Search for records.

        This operation converts a query to a vector embedding and then searches a namespace.
        You can optionally provide a reranking operation as part of the search.

        Args:
            namespace: The namespace in the index to search.
            query: The SearchQuery to use for the search. The query can include a
                ``match_terms`` field to specify which terms must be present in the text
                of each search hit. The match_terms should be a dict with ``strategy``
                (str) and ``terms`` (list[str]) keys, e.g.
                ``{"strategy": "all", "terms": ["term1", "term2"]}``. Currently only
                "all" strategy is supported, which means all specified terms must be
                present. **Note:** match_terms is only supported for sparse indexes with
                integrated embedding configured to use the pinecone-sparse-english-v0
                model.
            rerank: The SearchRerank to use with the search request. [optional]
            fields: List of fields to return in the response. Defaults to ["*"] which
                returns all fields. [optional]

        Returns:
            SearchRecordsResponse containing the records that match the search.

        Raises:
            Exception: If namespace is not provided.

        Examples:
            >>> from pinecone import SearchQuery, SearchRerank, RerankModel
            >>> await index.record.search(
            ...     namespace='my-namespace',
            ...     query=SearchQuery(
            ...         inputs={
            ...             "text": "Apple corporation",
            ...         },
            ...         top_k=3,
            ...     ),
            ...     rerank=SearchRerank(
            ...         model=RerankModel.Bge_Reranker_V2_M3,
            ...         rank_fields=["my_text_field"],
            ...         top_n=3,
            ...     ),
            ... )
        """
        if namespace is None:
            raise Exception("Namespace is required when searching records")

        request = IndexRequestFactory.search_request(query=query, rerank=rerank, fields=fields)

        from typing import cast

        result = await self._vector_api.search_records_namespace(namespace, request)
        return cast(SearchRecordsResponse, result)

    @validate_and_convert_errors
    async def search_records(
        self,
        namespace: str,
        query: SearchQueryTypedDict | SearchQuery,
        rerank: (SearchRerankTypedDict | SearchRerank) | None = None,
        fields: list[str] | None = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """Search for records (alias for search method).

        This is an alias for the ``search`` method. See :meth:`search` for full
        documentation.

        Args:
            namespace: The namespace in the index to search.
            query: The SearchQuery to use for the search.
            rerank: The SearchRerank to use with the search request. [optional]
            fields: List of fields to return in the response. Defaults to ["*"] which
                returns all fields. [optional]

        Returns:
            SearchRecordsResponse containing the records that match the search.
        """
        return await self.search(namespace, query=query, rerank=rerank, fields=fields)
