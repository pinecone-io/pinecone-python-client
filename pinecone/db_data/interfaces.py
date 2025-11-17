from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Literal

from pinecone.core.openapi.db_data.models import (
    IndexDescription as DescribeIndexStatsResponse,
    ListResponse,
    SearchRecordsResponse,
    NamespaceDescription,
    ListNamespacesResponse,
)
from .query_results_aggregator import QueryNamespacesResults
from multiprocessing.pool import ApplyResult
from .types import (
    VectorTypedDict,
    SparseVectorTypedDict,
    VectorMetadataTypedDict,
    FilterTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
    SearchQueryTypedDict,
    SearchRerankTypedDict,
)
from .dataclasses import (
    SearchQuery,
    SearchRerank,
    FetchResponse,
    FetchByMetadataResponse,
    QueryResponse,
    UpsertResponse,
    UpdateResponse,
    SparseValues,
    Vector,
)
from pinecone.utils import require_kwargs


class IndexInterface(ABC):
    @abstractmethod
    def upsert(
        self,
        vectors: (
            list[Vector] | list[VectorTuple] | list[VectorTupleWithMetadata] | list[VectorTypedDict]
        ),
        namespace: str | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse | ApplyResult:
        """
        Args:
            vectors (Union[list[Vector], list[VectorTuple], list[VectorTupleWithMetadata], list[VectorTypedDict]]): A list of vectors to upsert.
            namespace (str): The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size (int): The number of vectors to upsert in each batch.
                               If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress (bool): Whether to show a progress bar using tqdm.
                                  Applied only if batch_size is provided. Default is True.

        Returns:
            `UpsertResponse`, includes the number of vectors upserted.


        The upsert operation writes vectors into a namespace of your index.

        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        To upsert in parallel follow `this link <https://docs.pinecone.io/docs/insert-data#sending-upserts-in-parallel>`_.

        **Upserting dense vectors**

        When working with dense vectors, the dimension of each vector must match the dimension configured for the
        index.

        A vector can be represented in a variety of ways.

        .. code-block:: python
            :caption: Upserting a dense vector using the Vector object
            :emphasize-lines: 9-13

            from pinecone import Pinecone, Vector

            pc = Pinecone()
            idx = pc.Index(name="index-name")

            idx.upsert(
                namespace = 'my-namespace',
                vectors = [
                    Vector(
                        id='id1',
                        values=[0.1, 0.2, 0.3, 0.4],
                        metadata={'metadata_key': 'metadata_value'}
                    ),
                ]
            )

        .. code-block:: python
            :caption: Upserting a dense vector as a two-element tuple (no metadata)
            :emphasize-lines: 4

            idx.upsert(
                namespace = 'my-namespace',
                vectors = [
                    ('id1', [0.1, 0.2, 0.3, 0.4]),
                ]
            )

        .. code-block:: python
            :caption: Upserting a dense vector as a three-element tuple with metadata
            :emphasize-lines: 4-8

            idx.upsert(
                namespace = 'my-namespace',
                vectors = [
                    (
                        'id1',
                        [0.1, 0.2, 0.3, 0.4],
                        {'metadata_key': 'metadata_value'}
                    ),
                ]
            )

        .. code-block:: python
            :caption: Upserting a dense vector using a vector dictionary
            :emphasize-lines: 4-8

            idx.upsert(
                namespace = 'my-namespace',
                vectors = [
                    {
                        "id": 1,
                        "values": [0.1, 0.2, 0.3, 0.4],
                        "metadata": {"metadata_key": "metadata_value"}
                    },
                ]

        **Upserting sparse vectors**

        .. code-block:: python
            :caption: Upserting a sparse vector
            :emphasize-lines: 32-38

            from pinecone import (
                Pinecone,
                Metric,
                Vector,
                SparseValues,
                VectorType,
                ServerlessSpec,
                CloudProvider,
                AwsRegion
            )

            pc = Pinecone() # Reads PINECONE_API_KEY from environment variable

            # Create a sparse index
            index_description = pc.create_index(
                name="example-sparse",
                metric=Metric.Dotproduct,
                vector_type=VectorType.Sparse,
                spec=ServerlessSpec(
                    cloud=CloudProvider.AWS,
                    region=AwsRegion.US_WEST_2,
                )
            )

            # Target the index created above
            idx = pc.Index(host=index_description.host)

            # Upsert a sparse vector
            idx.upsert(
                namespace='my-namespace',
                vectors=[
                    Vector(
                        id='id1',
                        sparse_values=SparseValues(
                            indices=[1, 2],
                            values=[0.2, 0.4]
                        )
                    ),
                ]
            )

        .. code-block:: python
            :caption: Upserting a sparse vector using a dictionary
            :emphasize-lines: 4-10

            idx.upsert(
                namespace = 'my-namespace',
                vectors = [
                    {
                        "id": 1,
                        "sparse_values": {
                            "indices": [1, 2],
                            "values": [0.2, 0.4]
                        }
                    },
                ]
            )


        **Batch upsert**

        If you have a large number of vectors, you can upsert them in batches.

        .. code-block:: python
            :caption: Upserting in batches
            :emphasize-lines: 19

            from pinecone import Pinecone, Vector
            import random

            pc = Pinecone()
            idx = pc.Index(host="example-index-dojoi3u.svc.preprod-aws-0.pinecone.io")

            # Create some fake vector data for demonstration
            num_vectors = 100000
            vectors = [
                Vector(
                    id=f'id{i}',
                    values=[random.random() for _ in range(1536)])
                for i in range(num_vectors)
            ]

            idx.upsert(
                namespace='my-namespace',
                vectors=vectors,
                batch_size=50
            )


        **Visual progress bar with tqdm**

        To see a progress bar when upserting in batches, you will need to separately install `tqdm <https://tqdm.github.io/>`_.
        If ``tqdm`` is present, the client will detect and use it to display progress when ``show_progress=True``.


        """
        pass

    @abstractmethod
    def upsert_from_dataframe(
        self, df, namespace: str | None = None, batch_size: int = 500, show_progress: bool = True
    ):
        """Upsert vectors from a pandas DataFrame into the index.

        Args:
            df: A pandas DataFrame with the following columns: id, values, sparse_values, and metadata.
            namespace: The namespace to upsert into. If not specified, the default namespace is used. [optional]
            batch_size: The number of rows to upsert in a single batch. Defaults to 500.
            show_progress: Whether to show a progress bar. Defaults to True.

        Returns:
            UpsertResponse: Object containing the number of vectors upserted.

        Examples:

        .. code-block:: python

            import pandas as pd
            from pinecone import Pinecone

            pc = Pinecone()
            idx = pc.Index(host="example-index-host")

            # Create a DataFrame with vector data
            df = pd.DataFrame({
                'id': ['id1', 'id2', 'id3'],
                'values': [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9]
                ],
                'metadata': [
                    {'key1': 'value1'},
                    {'key2': 'value2'},
                    {'key3': 'value3'}
                ]
            })

            # Upsert from DataFrame
            response = idx.upsert_from_dataframe(
                df=df,
                namespace='my-namespace',
                batch_size=100,
                show_progress=True
            )

        """
        pass

    @abstractmethod
    def upsert_records(self, namespace: str, records: list[dict]) -> UpsertResponse:
        """Upsert records to a namespace.

        A record is a dictionary that contains either an ``id`` or ``_id``
        field along with other fields that will be stored as metadata. The ``id`` or ``_id`` field is used
        as the unique identifier for the record. At least one field in the record should correspond to
        a field mapping in the index's embed configuration.

        When records are upserted, Pinecone converts mapped fields into embeddings and upserts them into
        the specified namespace of the index.

        Args:
            namespace: The namespace of the index to upsert records to.
            records: The records to upsert into the index. Each record should contain an ``id`` or ``_id``
                    field and fields that match the index's embed configuration field mappings.

        Returns:
            UpsertResponse: Object which contains the number of records upserted.

        Examples:

        .. code-block:: python
            :caption: Upserting records to be embedded with Pinecone's integrated inference models

            from pinecone import (
                Pinecone,
                CloudProvider,
                AwsRegion,
                EmbedModel,
                IndexEmbed
            )

            pc = Pinecone(api_key="<<PINECONE_API_KEY>>")

            # Create an index configured for the multilingual-e5-large model
            index_model = pc.create_index_for_model(
                name="my-model-index",
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_WEST_2,
                embed=IndexEmbed(
                    model=EmbedModel.Multilingual_E5_Large,
                    field_map={"text": "my_text_field"}
                )
            )

            # Instantiate the index client
            idx = pc.Index(host=index_model.host)

            # Upsert records
            idx.upsert_records(
                namespace="my-namespace",
                records=[
                    {
                        "_id": "test1",
                        "my_text_field": "Apple is a popular fruit known for its sweetness and crisp texture.",
                    },
                    {
                        "_id": "test2",
                        "my_text_field": "The tech company Apple is known for its innovative products like the iPhone.",
                    },
                    {
                        "_id": "test3",
                        "my_text_field": "Many people enjoy eating apples as a healthy snack.",
                    },
                ],
            )

        """
        pass

    @abstractmethod
    def search(
        self,
        namespace: str,
        query: SearchQueryTypedDict | SearchQuery,
        rerank: (SearchRerankTypedDict | SearchRerank) | None = None,
        fields: list[str] | None = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """Search for records in a namespace.

        This operation converts a query to a vector embedding and then searches a namespace. You
        can optionally provide a reranking operation as part of the search.

        Args:
            namespace: The namespace in the index to search.
            query: The SearchQuery to use for the search. The query can include a ``match_terms`` field
                   to specify which terms must be present in the text of each search hit. The match_terms
                   should be a dict with ``strategy`` (str) and ``terms`` (list[str]) keys, e.g.
                   ``{"strategy": "all", "terms": ["term1", "term2"]}``. Currently only "all" strategy
                   is supported, which means all specified terms must be present.
                   **Note:** match_terms is only supported for sparse indexes with integrated embedding
                   configured to use the pinecone-sparse-english-v0 model.
            rerank: The SearchRerank to use with the search request. [optional]
            fields: List of fields to return in the response. Defaults to ["*"] to return all fields. [optional]

        Returns:
            SearchRecordsResponse: The records that match the search.

        Examples:

        .. code-block:: python

            from pinecone import (
                Pinecone,
                CloudProvider,
                AwsRegion,
                EmbedModel,
                IndexEmbed,
                SearchQuery,
                SearchRerank,
                RerankModel
            )

            pc = Pinecone(api_key="<<PINECONE_API_KEY>>")

            # Create an index for your embedding model
            index_model = pc.create_index_for_model(
                name="my-model-index",
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_WEST_2,
                embed=IndexEmbed(
                    model=EmbedModel.Multilingual_E5_Large,
                    field_map={"text": "my_text_field"}
                )
            )

            # Instantiate the index client
            idx = pc.Index(host=index_model.host)

            # Search for similar records
            response = idx.search(
                namespace="my-namespace",
                query=SearchQuery(
                    inputs={
                        "text": "Apple corporation",
                    },
                    top_k=3,
                ),
                rerank=SearchRerank(
                    model=RerankModel.Bge_Reranker_V2_M3,
                    rank_fields=["my_text_field"],
                    top_n=3,
                ),
            )

        """
        pass

    @abstractmethod
    def search_records(
        self,
        namespace: str,
        query: SearchQueryTypedDict | SearchQuery,
        rerank: (SearchRerankTypedDict | SearchRerank) | None = None,
        fields: list[str] | None = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """Alias of the search() method.

        See :meth:`search` for full documentation and examples.

        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: list[str] | None = None,
        delete_all: bool | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Delete vectors from the index, from a single namespace.

        The Delete operation deletes vectors from the index, from a single namespace.
        No error is raised if the vector id does not exist.

        Note: For any delete call, if namespace is not specified, the default namespace ``""`` is used.
        Since the delete operation does not error when ids are not present, this means you may not receive
        an error if you delete from the wrong namespace.

        Delete can occur in the following mutually exclusive ways:

        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
           (note that for this option delete_all must be set to False)

        Args:
            ids: Vector ids to delete. [optional]
            delete_all: This indicates that all vectors in the index namespace should be deleted.
                       Default is False. [optional]
            namespace: The namespace to delete vectors from. If not specified, the default namespace is used. [optional]
            filter: If specified, the metadata filter here will be used to select the vectors to delete.
                   This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                   See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            dict[str, Any]: An empty dictionary if the delete operation was successful.

        Examples:

        .. code-block:: python

            >>> # Delete specific vectors by ID
            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            {}

            >>> # Delete all vectors from a namespace
            >>> index.delete(delete_all=True, namespace='my_namespace')
            {}

            >>> # Delete vectors matching a metadata filter
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace')
            {}

        """
        pass

    @abstractmethod
    def fetch(self, ids: list[str], namespace: str | None = None, **kwargs) -> FetchResponse:
        """Fetch vectors by ID from a single namespace.

        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Args:
            ids: The vector IDs to fetch.
            namespace: The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            FetchResponse: Object which contains the list of Vector objects, and namespace name.

        Examples:

        .. code-block:: python

            >>> # Fetch vectors from a specific namespace
            >>> response = index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> for vector_id, vector in response.vectors.items():
            ...     print(f"{vector_id}: {vector.values}")

            >>> # Fetch vectors from the default namespace
            >>> response = index.fetch(ids=['id1', 'id2'])

        """
        pass

    @abstractmethod
    def fetch_by_metadata(
        self,
        filter: FilterTypedDict,
        namespace: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        **kwargs,
    ) -> FetchByMetadataResponse:
        """
        Fetch vectors by metadata filter.

        Look up and return vectors by metadata filter from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Examples:

        .. code-block:: python

            >>> index.fetch_by_metadata(
            ...     filter={'genre': {'$in': ['comedy', 'drama']}, 'year': {'$eq': 2019}},
            ...     namespace='my_namespace',
            ...     limit=50
            ... )
            >>> index.fetch_by_metadata(
            ...     filter={'status': 'active'},
            ...     pagination_token='token123'
            ... )

        Args:
            filter (dict[str, Union[str, float, int, bool, List, dict]]):
                Metadata filter expression to select vectors.
                See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`
            namespace (str): The namespace to fetch vectors from.
                            If not specified, the default namespace is used. [optional]
            limit (int): Max number of vectors to return. Defaults to 100. [optional]
            pagination_token (str): Pagination token to continue a previous listing operation. [optional]

        Returns:
            FetchByMetadataResponse: Object containing the fetched vectors, namespace, usage, and pagination token.
        """
        pass

    @abstractmethod
    def query(
        self,
        *args,
        top_k: int,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: (SparseValues | SparseVectorTypedDict) | None = None,
        **kwargs,
    ) -> QueryResponse | ApplyResult:
        """
        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        Examples:

        .. code-block:: python

            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> index.query(id='id1', top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace', filter={'key': 'value'})
            >>> index.query(id='id1', top_k=10, namespace='my_namespace', include_metadata=True, include_values=True)
            >>> index.query(vector=[1, 2, 3], sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>             top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], sparse_vector=SparseValues([1, 2], [0.2, 0.4]),
            >>>             top_k=10, namespace='my_namespace')

        Args:
            vector (list[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each `query()` request can contain only one of the parameters
                                  `id` or `vector`.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each `query()` request can contain only one of the parameters
                      `vector` or  `id`. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to query vectors from.
                             If not specified, the default namespace is used. [optional]
            filter: The filter to apply. You can use vector metadata to limit your search.
                   See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]
            include_values (bool): Indicates whether vector values are included in the response.
                                   If omitted the server will use the default value of False [optional]
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
                                     If omitted the server will use the default value of False  [optional]
            sparse_vector: (Union[SparseValues, dict[str, Union[list[float], list[int]]]]): sparse values of the query vector.
                            Expected to be either a SparseValues object or a dict of the form:
                             {'indices': list[int], 'values': list[float]}, where the lists each have the same length.

        Returns: QueryResponse object which contains the list of the closest vectors as ScoredVector objects,
                 and namespace name.
        """
        pass

    @abstractmethod
    def query_namespaces(
        self,
        vector: list[float] | None,
        namespaces: list[str],
        metric: Literal["cosine", "euclidean", "dotproduct"],
        top_k: int | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: (SparseValues | SparseVectorTypedDict) | None = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        """The ``query_namespaces()`` method is used to make a query to multiple namespaces in parallel and combine the results into one result set.

        :param vector: The query vector, must be the same length as the dimension of the index being queried.
        :type vector: list[float]
        :param namespaces: The list of namespaces to query.
        :type namespaces: list[str]
        :param top_k: The number of results you would like to request from each namespace. Defaults to 10.
        :type top_k: Optional[int]
        :param metric: Must be one of 'cosine', 'euclidean', 'dotproduct'. This is needed in order to merge results across namespaces, since the interpretation of score depends on the index metric type.
        :type metric: str
        :param filter: Pass an optional filter to filter results based on metadata. Defaults to None.
        :type filter: Optional[dict[str, Union[str, float, int, bool, List, dict]]]
        :param include_values: Boolean field indicating whether vector values should be included with results. Defaults to None.
        :type include_values: Optional[bool]
        :param include_metadata: Boolean field indicating whether vector metadata should be included with results. Defaults to None.
        :type include_metadata: Optional[bool]
        :param sparse_vector: If you are working with a dotproduct index, you can pass a sparse vector as part of your hybrid search. Defaults to None.
        :type sparse_vector: Optional[ Union[SparseValues, dict[str, Union[list[float], list[int]]]] ]
        :return: A QueryNamespacesResults object containing the combined results from all namespaces, as well as the combined usage cost in read units.
        :rtype: QueryNamespacesResults

        .. admonition:: Note

            Since several asynchronous calls are made on your behalf when calling this method, you will need to tune
            the **pool_threads** and **connection_pool_maxsize** parameter of the Index constructor to suite your workload.
            If these values are too small in relation to your workload, you will experience performance issues as
            requests queue up while waiting for a request thread to become available.

        Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            index = pc.Index(
                host="index-name",
                pool_threads=32,
                connection_pool_maxsize=32
            )

            query_vec = [0.1, 0.2, 0.3] # An embedding that matches the index dimension
            combined_results = index.query_namespaces(
                vector=query_vec,
                namespaces=['ns1', 'ns2', 'ns3', 'ns4'],
                metric="cosine",
                top_k=10,
                filter={'genre': {"$eq": "drama"}},
                include_values=True,
                include_metadata=True
            )

            for vec in combined_results.matches:
                print(vec.id, vec.score)
            print(combined_results.usage)

        """
        pass

    @abstractmethod
    def update(
        self,
        id: str | None = None,
        values: list[float] | None = None,
        set_metadata: VectorMetadataTypedDict | None = None,
        namespace: str | None = None,
        sparse_values: (SparseValues | SparseVectorTypedDict) | None = None,
        filter: FilterTypedDict | None = None,
        dry_run: bool | None = None,
        **kwargs,
    ) -> UpdateResponse:
        """
        The Update operation updates vectors in a namespace.

        This method supports two update modes:

        1. **Single vector update by ID**: Provide `id` to update a specific vector.
           - Updates the vector with the given ID
           - If `values` is included, it will overwrite the previous vector values
           - If `set_metadata` is included, the metadata will be merged with existing metadata on the vector.
             Fields specified in `set_metadata` will overwrite existing fields with the same key, while
             fields not in `set_metadata` will remain unchanged.

        2. **Bulk update by metadata filter**: Provide `filter` to update all vectors matching the filter criteria.
           - Updates all vectors in the namespace that match the filter expression
           - Useful for updating metadata across multiple vectors at once
           - If `set_metadata` is included, the metadata will be merged with existing metadata on each vector.
             Fields specified in `set_metadata` will overwrite existing fields with the same key, while
             fields not in `set_metadata` will remain unchanged.
           - The response includes `matched_records` indicating how many vectors were updated

        Either `id` or `filter` must be provided (but not both in the same call).

        Examples:

        **Single vector update by ID:**

        .. code-block:: python

            >>> # Update vector values
            >>> index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> # Update vector metadata
            >>> index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace')
            >>> # Update vector values and sparse values
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>              namespace='my_namespace')
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]),
            >>>              namespace='my_namespace')

        **Bulk update by metadata filter:**

        .. code-block:: python

            >>> # Update metadata for all vectors matching the filter
            >>> response = index.update(set_metadata={'status': 'active'}, filter={'genre': {'$eq': 'drama'}},
            >>>                        namespace='my_namespace')
            >>> print(f"Updated {response.matched_records} vectors")
            >>> # Preview how many vectors would be updated (dry run)
            >>> response = index.update(set_metadata={'status': 'active'}, filter={'genre': {'$eq': 'drama'}},
            >>>                        namespace='my_namespace', dry_run=True)
            >>> print(f"Would update {response.matched_records} vectors")

        Args:
            id: Vector unique id. Required for single vector updates. Must not be provided when using filter. [optional]
            values: Vector values to set. [optional]
            set_metadata: Metadata to merge with existing metadata on the vector(s). Fields specified will overwrite
                         existing fields with the same key, while fields not specified will remain unchanged. [optional]
            namespace: Namespace name where to update the vector(s). [optional]
            sparse_values: Sparse values to update for the vector. Expected to be either a SparseValues object or a dict
                          of the form: ``{'indices': list[int], 'values': list[float]}`` where the lists each have
                          the same length. [optional]
            filter: A metadata filter expression. When provided, updates all vectors in the namespace that match
                   the filter criteria. See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`.
                   Must not be provided when using id. Either ``id`` or ``filter`` must be provided. [optional]
            dry_run: If ``True``, return the number of records that match the ``filter`` without executing
                    the update. Only meaningful when using ``filter`` (not with ``id``). Useful for previewing
                    the impact of a bulk update before applying changes. Defaults to ``False``. [optional]

        Returns:
            UpdateResponse: An UpdateResponse object. When using filter-based updates, the response includes
            `matched_records` indicating the number of vectors that were updated (or would be updated if
            `dry_run=True`).
        """
        pass

    @abstractmethod
    def describe_index_stats(
        self, filter: FilterTypedDict | None = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        """
        The DescribeIndexStats operation returns statistics about the index contents.
        For example: The vector count per namespace and the number of dimensions.

        Args:
            filter (dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.

        .. code-block:: python

            >>> pc = Pinecone()
            >>> index = pc.Index(name="my-index")
            >>> index.describe_index_stats()
            {'dimension': 1536,
            'index_fullness': 0.0,
            'metric': 'cosine',
            'namespaces': {'ns0': {'vector_count': 700},
                            'ns1': {'vector_count': 700},
                            'ns2': {'vector_count': 500},
                            'ns3': {'vector_count': 100},
                            'ns4': {'vector_count': 100},
                            'ns5': {'vector_count': 50},
                            'ns6': {'vector_count': 50}},
            'total_vector_count': 2200,
            'vector_type': 'dense'}

        """
        pass

    @abstractmethod
    def list_paginated(
        self,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str | None = None,
        **kwargs,
    ) -> ListResponse:
        """
        The list_paginated operation finds vectors based on an id prefix within a single namespace.
        It returns matching ids in a paginated form, with a pagination token to fetch the next page of results.
        This id list can then be passed to fetch or delete operations, depending on your use case.

        Consider using the `list` method to avoid having to handle pagination tokens manually.

        Examples:

        .. code-block:: python

            >>> results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace')
            >>> [v.id for v in results.vectors]
            ['99', '990', '991', '992', '993']
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> next_results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace', pagination_token=results.pagination.next)

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]

        Returns: ListResponse object which contains the list of ids, the namespace name, pagination information, and usage showing the number of read_units consumed.
        """
        pass

    @abstractmethod
    def list(self, **kwargs):
        """
        The list operation accepts all of the same arguments as list_paginated, and returns a generator that yields
        a list of the matching vector ids in each page of results. It automatically handles pagination tokens on your
        behalf.

        Examples:
            >>> for ids in index.list(prefix='99', limit=5, namespace='my_namespace'):
            >>>     print(ids)
            ['99', '990', '991', '992', '993']
            ['994', '995', '996', '997', '998']
            ['999']

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]
        """
        pass

    @abstractmethod
    @require_kwargs
    def create_namespace(
        self, name: str, schema: dict[str, Any] | None = None, **kwargs
    ) -> NamespaceDescription:
        """Create a namespace in a serverless index.

        Args:
            name (str): The name of the namespace to create
            schema (Optional[dict[str, Any]]): Optional schema configuration for the namespace as a dictionary. [optional]

        Returns:
            NamespaceDescription: Information about the created namespace including vector count

        Create a namespace in a serverless index. For guidance and examples, see
        `Manage namespaces <https://docs.pinecone.io/guides/manage-data/manage-namespaces>`_.

        **Note:** This operation is not supported for pod-based indexes.

        Examples:

            .. code-block:: python

                >>> # Create a namespace with just a name
                >>> namespace = index.create_namespace(name="my-namespace")
                >>> print(f"Created namespace: {namespace.name}, Vector count: {namespace.vector_count}")

                >>> # Create a namespace with schema configuration
                >>> from pinecone.core.openapi.db_data.model.create_namespace_request_schema import CreateNamespaceRequestSchema
                >>> schema = CreateNamespaceRequestSchema(fields={...})
                >>> namespace = index.create_namespace(name="my-namespace", schema=schema)
        """
        pass

    @abstractmethod
    @require_kwargs
    def describe_namespace(self, namespace: str, **kwargs) -> NamespaceDescription:
        """Describe a namespace within an index, showing the vector count within the namespace.

        Args:
            namespace (str): The namespace to describe

        Returns:
            NamespaceDescription: Information about the namespace including vector count
        """
        pass

    @abstractmethod
    @require_kwargs
    def delete_namespace(self, namespace: str, **kwargs) -> dict[str, Any]:
        """Delete a namespace from an index.

        Args:
            namespace: The namespace to delete.

        Returns:
            dict[str, Any]: Response from the delete operation.
        """
        pass

    @abstractmethod
    @require_kwargs
    def list_namespaces(
        self, limit: int | None = None, **kwargs
    ) -> Iterator[ListNamespacesResponse]:
        """List all namespaces in an index. This method automatically handles pagination to return all results.

        Args:
            limit: The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]

        Returns:
            Iterator[ListNamespacesResponse]: An iterator that yields ListNamespacesResponse objects containing the list of namespaces.

        Examples:

        .. code-block:: python

            >>> # Iterate over all namespaces
            >>> for namespace_response in index.list_namespaces(limit=5):
            ...     for namespace in namespace_response.namespaces:
            ...         print(f"Namespace: {namespace.name}, Vector count: {namespace.vector_count}")

            >>> # Convert to list (be cautious with large datasets)
            >>> results = list(index.list_namespaces(limit=5))
            >>> for namespace_response in results:
            ...     for namespace in namespace_response.namespaces:
            ...         print(f"Namespace: {namespace.name}, Vector count: {namespace.vector_count}")

        """
        pass

    @abstractmethod
    @require_kwargs
    def list_namespaces_paginated(
        self, limit: int | None = None, pagination_token: str | None = None, **kwargs
    ) -> ListNamespacesResponse:
        """List all namespaces in an index with pagination support. The response includes pagination information if there are more results available.

        Consider using the ``list_namespaces`` method to avoid having to handle pagination tokens manually.

        Args:
            limit: The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]
            pagination_token: A token needed to fetch the next page of results. This token is returned
                            in the response if additional results are available. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            ListNamespacesResponse: Object containing the list of namespaces and pagination information.

        Examples:

        .. code-block:: python

            >>> # Get first page of namespaces
            >>> results = index.list_namespaces_paginated(limit=5)
            >>> for namespace in results.namespaces:
            ...     print(f"Namespace: {namespace.name}, Vector count: {namespace.vector_count}")

            >>> # Get next page if available
            >>> if results.pagination and results.pagination.next:
            ...     next_results = index.list_namespaces_paginated(
            ...         limit=5,
            ...         pagination_token=results.pagination.next
            ...     )

        """
        pass
