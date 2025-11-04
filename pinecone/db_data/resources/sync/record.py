from typing import Union, List, Optional, Dict, Any
import logging

from pinecone.core.openapi.db_data.models import SearchRecordsResponse
from pinecone.db_data.request_factory import IndexRequestFactory
from pinecone.db_data.types import (
    SearchQueryTypedDict,
    SearchRerankTypedDict,
)
from pinecone.db_data.dataclasses import SearchQuery, SearchRerank
from pinecone.utils import validate_and_convert_errors, filter_dict
from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS

logger = logging.getLogger(__name__)
""" :meta private: """


class RecordResource:
    """Resource for record-based operations on a Pinecone index."""

    def __init__(self, vector_api, openapi_config):
        self._vector_api = vector_api
        """ :meta private: """
        self._openapi_config = openapi_config
        """ :meta private: """

    def _openapi_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return filter_dict(kwargs, OPENAPI_ENDPOINT_PARAMS)

    @validate_and_convert_errors
    def search(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        request = IndexRequestFactory.search_request(query=query, rerank=rerank, fields=fields)

        return self._vector_api.search_records_namespace(namespace, request)

    @validate_and_convert_errors
    def search_records(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        return self.search(namespace, query=query, rerank=rerank, fields=fields)

    @validate_and_convert_errors
    def upsert_records(self, namespace: str, records: List[Dict[str, Any]]):
        """Upsert records to a namespace.

        Upsert records to a namespace. A record is a dictionary that contains either an ``id`` or ``_id``
        field along with other fields that will be stored as metadata. The ``id`` or ``_id`` field is used
        as the unique identifier for the record. At least one field in the record should correspond to
        a field mapping in the index's embed configuration.

        When records are upserted, Pinecone converts mapped fields into embeddings and upserts them into
        the specified namespace of the index.

        Args:
            namespace (str): The namespace of the index to upsert records to.
            records (List[Dict[str, Any]]): The records to upsert into the index.
                Each record should contain either an ``id`` or ``_id`` field.

        Examples:

        .. code-block:: python

            >>> from pinecone import Pinecone, CloudProvider, AwsRegion, EmbedModel, IndexEmbed
            >>> pc = Pinecone(api_key="<<PINECONE_API_KEY>>")
            >>> index_model = pc.create_index_for_model(
            ...     name="my-model-index",
            ...     cloud=CloudProvider.AWS,
            ...     region=AwsRegion.US_WEST_2,
            ...     embed=IndexEmbed(
            ...         model=EmbedModel.Multilingual_E5_Large,
            ...         field_map={"text": "my_text_field"}
            ...     )
            ... )
            >>> idx = pc.Index(host=index_model.host)
            >>> idx.upsert_records(
            ...     namespace="my-namespace",
            ...     records=[
            ...         {"_id": "test1", "my_text_field": "Apple is a popular fruit."},
            ...         {"_id": "test2", "my_text_field": "The tech company Apple is innovative."},
            ...     ],
            ... )
        """
        args = IndexRequestFactory.upsert_records_args(namespace=namespace, records=records)
        self._vector_api.upsert_records_namespace(**args)

