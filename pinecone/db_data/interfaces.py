from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict, Any, Iterator

from pinecone.core.openapi.db_data.models import (
    FetchResponse,
    QueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
    UpsertResponse,
    Vector,
    ListResponse,
    SparseValues,
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
from .dataclasses import SearchQuery, SearchRerank
from pinecone.utils import require_kwargs


class IndexInterface(ABC):
    @abstractmethod
    def upsert(
        self,
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
        ],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse:
        """
        Args:
            vectors (Union[List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]]): A list of vectors to upsert.
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
        self, df, namespace: Optional[str] = None, batch_size: int = 500, show_progress: bool = True
    ):
        """Upserts a dataframe into the index.

        Args:
            df: A pandas dataframe with the following columns: id, values, sparse_values, and metadata.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
            show_progress: Whether to show a progress bar.
        """
        pass

    @abstractmethod
    def upsert_records(self, namespace: str, records: List[Dict]):
        """
        :param namespace: The namespace of the index to upsert records to.
        :type namespace: str, required
        :param records: The records to upsert into the index.
        :type records: List[Dict], required
        :return: UpsertResponse object which contains the number of records upserted.

        Upsert records to a namespace. A record is a dictionary that contains eitiher an `id` or `_id`
        field along with other fields that will be stored as metadata. The `id` or `_id` field is used
        as the unique identifier for the record. At least one field in the record should correspond to
        a field mapping in the index's embed configuration.

        When records are upserted, Pinecone converts mapped fields into embeddings and upserts them into
        the specified namespacce of the index.

        .. code-block:: python
            :caption: Upserting records to be embedded with Pinecone's integrated inference models

            from pinecone import (
                Pinecone,
                CloudProvider,
                AwsRegion,
                EmbedModel
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

            # upsert records
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
                    {
                        "_id": "test4",
                        "my_text_field": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
                    },
                    {
                        "_id": "test5",
                        "my_text_field": "An apple a day keeps the doctor away, as the saying goes.",
                    },
                    {
                        "_id": "test6",
                        "my_text_field": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership.",
                    },
                ],
            )

            from pinecone import SearchQuery, SearchRerank, RerankModel

            # Search for similar records
            response = idx.search_records(
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
    def search(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """
        :param namespace: The namespace in the index to search.
        :type namespace: str, required
        :param query: The SearchQuery to use for the search.
        :type query: Union[Dict, SearchQuery], required
        :param rerank: The SearchRerank to use with the search request.
        :type rerank: Union[Dict, SearchRerank], optional
        :return: The records that match the search.

        Search for records.

        This operation converts a query to a vector embedding and then searches a namespace. You
        can optionally provide a reranking operation as part of the search.

        .. code-block:: python

            from pinecone import (
                Pinecone,
                CloudProvider,
                AwsRegion,
                EmbedModel
                IndexEmbed
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

            # upsert records
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
                    {
                        "_id": "test4",
                        "my_text_field": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
                    },
                    {
                        "_id": "test5",
                        "my_text_field": "An apple a day keeps the doctor away, as the saying goes.",
                    },
                    {
                        "_id": "test6",
                        "my_text_field": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership.",
                    },
                ],
            )

            from pinecone import SearchQuery, SearchRerank, RerankModel

            # search for similar records
            response = idx.search_records(
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
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """Alias of the search() method."""
        pass

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            ids (List[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                                Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                            If not specified, the default namespace is used.
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                    See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]


        The Delete operation deletes vectors from the index, from a single namespace.

        No error is raised if the vector id does not exist.

        Note: For any delete call, if namespace is not specified, the default namespace ``""`` is used.
        Since the delete operation does not error when ids are not present, this means you may not receive
        an error if you delete from the wrong namespace.

        Delete can occur in the following mutual exclusive ways:

        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
            (note that for this option delete all must be set to False)

        Examples:

        .. code-block:: python

            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.delete(delete_all=True, namespace='my_namespace')
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace')


        Returns: An empty dictionary if the delete operation was successful.
        """
        pass

    @abstractmethod
    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> FetchResponse:
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Examples:

        .. code-block:: python

            >>> index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.fetch(ids=['id1', 'id2'])

        Args:
            ids (List[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]

        Returns: FetchResponse object which contains the list of Vector objects, and namespace name.
        """
        pass

    @abstractmethod
    def query(
        self,
        *args,
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Union[QueryResponse, ApplyResult]:
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
            vector (List[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each `query()` request can contain only one of the parameters
                                  `id` or `vector`.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each `query()` request can contain only one of the parameters
                      `vector` or  `id`. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to query vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (Dict[str, Union[str, float, int, bool, List, dict]):
                    The filter to apply. You can use vector metadata to limit your search.
                    See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]
            include_values (bool): Indicates whether vector values are included in the response.
                                   If omitted the server will use the default value of False [optional]
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
                                     If omitted the server will use the default value of False  [optional]
            sparse_vector: (Union[SparseValues, Dict[str, Union[List[float], List[int]]]]): sparse values of the query vector.
                            Expected to be either a SparseValues object or a dict of the form:
                             {'indices': List[int], 'values': List[float]}, where the lists each have the same length.

        Returns: QueryResponse object which contains the list of the closest vectors as ScoredVector objects,
                 and namespace name.
        """
        pass

    @abstractmethod
    def query_namespaces(
        self,
        vector: List[float],
        namespaces: List[str],
        top_k: Optional[int] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        """The ``query_namespaces()`` method is used to make a query to multiple namespaces in parallel and combine the results into one result set.

        :param vector: The query vector, must be the same length as the dimension of the index being queried.
        :type vector: List[float]
        :param namespaces: The list of namespaces to query.
        :type namespaces: List[str]
        :param top_k: The number of results you would like to request from each namespace. Defaults to 10.
        :type top_k: Optional[int]
        :param metric: Must be one of 'cosine', 'euclidean', 'dotproduct'. This is needed in order to merge results across namespaces, since the interpretation of score depends on the index metric type.
        :type metric: str
        :param filter: Pass an optional filter to filter results based on metadata. Defaults to None.
        :type filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]]
        :param include_values: Boolean field indicating whether vector values should be included with results. Defaults to None.
        :type include_values: Optional[bool]
        :param include_metadata: Boolean field indicating whether vector metadata should be included with results. Defaults to None.
        :type include_metadata: Optional[bool]
        :param sparse_vector: If you are working with a dotproduct index, you can pass a sparse vector as part of your hybrid search. Defaults to None.
        :type sparse_vector: Optional[ Union[SparseValues, Dict[str, Union[List[float], List[int]]]] ]
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
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        The Update operation updates vector in a namespace.
        If a value is included, it will overwrite the previous value.
        If a set_metadata is included,
        the values of the fields specified in it will be added or overwrite the previous value.

        Examples:

        .. code-block:: python

            >>> index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace')
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>              namespace='my_namespace')
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]),
            >>>              namespace='my_namespace')

        Args:
            id (str): Vector's unique id.
            values (List[float]): vector values to set. [optional]
            set_metadata (Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]]):
                metadata to set for vector. [optional]
            namespace (str): Namespace name where to update the vector.. [optional]
            sparse_values: (Dict[str, Union[List[float], List[int]]]): sparse values to update for the vector.
                           Expected to be either a SparseValues object or a dict of the form:
                           {'indices': List[int], 'values': List[float]} where the lists each have the same length.


        Returns: An empty dictionary if the update was successful.
        """
        pass

    @abstractmethod
    def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
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
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
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
    def delete_namespace(self, namespace: str, **kwargs) -> Dict[str, Any]:
        """Delete a namespace from an index.

        Args:
            namespace (str): The namespace to delete

        Returns:
            Dict[str, Any]: Response from the delete operation
        """
        pass

    @abstractmethod
    @require_kwargs
    def list_namespaces(
            self, limit: Optional[int] = None, **kwargs
    ) -> Iterator[ListNamespacesResponse]:
        """List all namespaces in an index. This method automatically handles pagination to return all results.

        Args:
            limit (Optional[int]): The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]

        Returns:
            ``ListNamespacesResponse``: Object containing the list of namespaces.

        Examples:
            .. code-block:: python
                >>> results = list(index.list_namespaces(limit=5))
                >>> for namespace in results:
                ...     print(f"Namespace: {namespace.name}, Vector count: {namespace.vector_count}")
                Namespace: namespace1, Vector count: 1000
                Namespace: namespace2, Vector count: 2000
        """
        pass

    @abstractmethod
    @require_kwargs
    def list_namespaces_paginated(
        self, limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> ListNamespacesResponse:
        """List all namespaces in an index with pagination support. The response includes pagination information if there are more results available.

        Consider using the ``list_namespaces`` method to avoid having to handle pagination tokens manually.

        Args:
            limit (Optional[int]): The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]

        Returns:
            ``ListNamespacesResponse``: Object containing the list of namespaces and pagination information.

        Examples:
            .. code-block:: python
                >>> results = index.list_namespaces_paginated(limit=5)
                >>> results.pagination.next
                eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
                >>> next_results = index.list_namespaces_paginated(limit=5, pagination_token=results.pagination.next)
        """
        pass