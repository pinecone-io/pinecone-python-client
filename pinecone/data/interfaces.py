from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict, Any

from pinecone.core.openapi.db_data.models import (
    FetchResponse,
    QueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
    UpsertResponse,
    Vector,
    ListResponse,
    SparseValues,
    SearchRecordsResponse,
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
        The upsert operation writes vectors into a namespace.
        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        To upsert in parallel follow: https://docs.pinecone.io/docs/insert-data#sending-upserts-in-parallel

        A vector can be represented by a 1) Vector object, a 2) tuple or 3) a dictionary

        If a tuple is used, it must be of the form `(id, values, metadata)` or `(id, values)`.
        where id is a string, vector is a list of floats, metadata is a dict,
        and sparse_values is a dict of the form `{'indices': List[int], 'values': List[float]}`.

        Examples:
            >>> ('id1', [1.0, 2.0, 3.0], {'key': 'value'}, {'indices': [1, 2], 'values': [0.2, 0.4]})
            >>> ('id1', [1.0, 2.0, 3.0], None, {'indices': [1, 2], 'values': [0.2, 0.4]})
            >>> ('id1', [1.0, 2.0, 3.0], {'key': 'value'}), ('id2', [1.0, 2.0, 3.0])

        If a Vector object is used, a Vector object must be of the form
        `Vector(id, values, metadata, sparse_values)`, where metadata and sparse_values are optional
        arguments.

        Examples:
            >>> Vector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'})
            >>> Vector(id='id2', values=[1.0, 2.0, 3.0])
            >>> Vector(id='id3', values=[1.0, 2.0, 3.0], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]))

        **Note:** the dimension of each vector must match the dimension of the index.

        If a dictionary is used, it must be in the form `{'id': str, 'values': List[float], 'sparse_values': {'indices': List[int], 'values': List[float]}, 'metadata': dict}`

        Examples:
            >>> index.upsert([('id1', [1.0, 2.0, 3.0], {'key': 'value'}), ('id2', [1.0, 2.0, 3.0])])
            >>>
            >>> index.upsert([{'id': 'id1', 'values': [1.0, 2.0, 3.0], 'metadata': {'key': 'value'}},
            >>>               {'id': 'id2', 'values': [1.0, 2.0, 3.0], 'sparse_values': {'indices': [1, 8], 'values': [0.2, 0.4]}])
            >>> index.upsert([Vector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
            >>>               Vector(id='id2', values=[1.0, 2.0, 3.0], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]))])

        API reference: https://docs.pinecone.io/reference/upsert

        Args:
            vectors (Union[List[Vector], List[Tuple]]): A list of vectors to upsert.
            namespace (str): The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size (int): The number of vectors to upsert in each batch.
                               If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress (bool): Whether to show a progress bar using tqdm.
                                  Applied only if batch_size is provided. Default is True.
        Keyword Args:
            Supports OpenAPI client keyword arguments. See pinecone.core.client.models.UpsertRequest for more details.

        Returns: UpsertResponse, includes the number of vectors upserted.
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
        Upsert records to a namespace.

        Converts records into embeddings and upserts them into a namespacce in the index.

        :param namespace: The namespace of the index to upsert records to.
        :type namespace: str, required
        :param records: The records to upsert into the index.
        :type records: List[Dict], required
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
        Search for records.

        This operation converts a query to a vector embedding and then searches a namespace. You
        can optionally provide a reranking operation as part of the search.

        :param namespace: The namespace in the index to search.
        :type namespace: str, required
        :param query: The SearchQuery to use for the search.
        :type query: Union[Dict, SearchQuery], required
        :param rerank: The SearchRerank to use with the search request.
        :type rerank: Union[Dict, SearchRerank], optional
        :return: The records that match the search.
        :rtype: RecordModel
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
        The Delete operation deletes vectors from the index, from a single namespace.
        No error raised if the vector id does not exist.
        Note: for any delete call, if namespace is not specified, the default namespace is used.

        Delete can occur in the following mutual exclusive ways:
        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
            (note that for this option delete all must be set to False)

        API reference: https://docs.pinecone.io/reference/delete_post

        Examples:
            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.delete(delete_all=True, namespace='my_namespace')
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace')

        Args:
            ids (List[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                                Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                            If not specified, the default namespace is used.
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Keyword Args:
          Supports OpenAPI client keyword arguments. See pinecone.core.client.models.DeleteRequest for more details.


          Returns: An empty dictionary if the delete operation was successful.
        """
        pass

    @abstractmethod
    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> FetchResponse:
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        API reference: https://docs.pinecone.io/reference/fetch

        Examples:
            >>> index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.fetch(ids=['id1', 'id2'])

        Args:
            ids (List[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
        Keyword Args:
            Supports OpenAPI client keyword arguments. See pinecone.core.client.models.FetchResponse for more details.


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

        API reference: https://docs.pinecone.io/reference/query

        Examples:
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
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (Dict[str, Union[str, float, int, bool, List, dict]):
                    The filter to apply. You can use vector metadata to limit your search.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
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
        """The query_namespaces() method is used to make a query to multiple namespaces in parallel and combine the results into one result set.

        Since several asynchronous calls are made on your behalf when calling this method, you will need to tune the pool_threads and connection_pool_maxsize parameter of the Index constructor to suite your workload.

        Examples:

        ```python
        from pinecone import Pinecone

        pc = Pinecone(api_key="your-api-key")
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
        ```

        Args:
            vector (List[float]): The query vector, must be the same length as the dimension of the index being queried.
            namespaces (List[str]): The list of namespaces to query.
            top_k (Optional[int], optional): The number of results you would like to request from each namespace. Defaults to 10.
            metric (str): Must be one of 'cosine', 'euclidean', 'dotproduct'. This is needed in order to merge results across namespaces, since the interpretation of score depends on the index metric type.
            filter (Optional[Dict[str, Union[str, float, int, bool, List, dict]]], optional): Pass an optional filter to filter results based on metadata. Defaults to None.
            include_values (Optional[bool], optional): Boolean field indicating whether vector values should be included with results. Defaults to None.
            include_metadata (Optional[bool], optional): Boolean field indicating whether vector metadata should be included with results. Defaults to None.
            sparse_vector (Optional[ Union[SparseValues, Dict[str, Union[List[float], List[int]]]] ], optional): If you are working with a dotproduct index, you can pass a sparse vector as part of your hybrid search. Defaults to None.

        Returns:
            QueryNamespacesResults: A QueryNamespacesResults object containing the combined results from all namespaces, as well as the combined usage cost in read units.
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

        API reference: https://docs.pinecone.io/reference/update

        Examples:
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

        Keyword Args:
            Supports OpenAPI client keyword arguments. See pinecone.core.client.models.UpdateRequest for more details.

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

        API reference: https://docs.pinecone.io/reference/describe_index_stats_post

        Examples:
            >>> index.describe_index_stats()
            >>> index.describe_index_stats(filter={'key': 'value'})

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.
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


class AsyncioIndexInterface(ABC):
    @abstractmethod
    async def upsert(
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
        The upsert operation writes vectors into a namespace.
        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        To upsert in parallel follow: https://docs.pinecone.io/docs/insert-data#sending-upserts-in-parallel

        A vector can be represented by a 1) Vector object, a 2) tuple or 3) a dictionary

        If a tuple is used, it must be of the form `(id, values, metadata)` or `(id, values)`.
        where id is a string, vector is a list of floats, metadata is a dict,
        and sparse_values is a dict of the form `{'indices': List[int], 'values': List[float]}`.

        Examples:
            >>> ('id1', [1.0, 2.0, 3.0], {'key': 'value'}, {'indices': [1, 2], 'values': [0.2, 0.4]})
            >>> ('id1', [1.0, 2.0, 3.0], None, {'indices': [1, 2], 'values': [0.2, 0.4]})
            >>> ('id1', [1.0, 2.0, 3.0], {'key': 'value'}), ('id2', [1.0, 2.0, 3.0])

        If a Vector object is used, a Vector object must be of the form
        `Vector(id, values, metadata, sparse_values)`, where metadata and sparse_values are optional
        arguments.

        Examples:
            >>> Vector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'})
            >>> Vector(id='id2', values=[1.0, 2.0, 3.0])
            >>> Vector(id='id3', values=[1.0, 2.0, 3.0], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]))

        **Note:** the dimension of each vector must match the dimension of the index.

        If a dictionary is used, it must be in the form `{'id': str, 'values': List[float], 'sparse_values': {'indices': List[int], 'values': List[float]}, 'metadata': dict}`

        Examples:
            >>> index.upsert([('id1', [1.0, 2.0, 3.0], {'key': 'value'}), ('id2', [1.0, 2.0, 3.0])])
            >>>
            >>> index.upsert([{'id': 'id1', 'values': [1.0, 2.0, 3.0], 'metadata': {'key': 'value'}},
            >>>               {'id': 'id2', 'values': [1.0, 2.0, 3.0], 'sparse_values': {'indices': [1, 8], 'values': [0.2, 0.4]}])
            >>> index.upsert([Vector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
            >>>               Vector(id='id2', values=[1.0, 2.0, 3.0], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]))])

        API reference: https://docs.pinecone.io/reference/upsert

        Args:
            vectors (Union[List[Vector], List[Tuple]]): A list of vectors to upsert.
            namespace (str): The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size (int): The number of vectors to upsert in each batch.
                               If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress (bool): Whether to show a progress bar using tqdm.
                                  Applied only if batch_size is provided. Default is True.
        Keyword Args:
            Supports OpenAPI client keyword arguments. See pinecone.core.client.models.UpsertRequest for more details.

        Returns: UpsertResponse, includes the number of vectors upserted.
        """
        pass

    @abstractmethod
    async def upsert_from_dataframe(
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
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        The Delete operation deletes vectors from the index, from a single namespace.
        No error raised if the vector id does not exist.
        Note: for any delete call, if namespace is not specified, the default namespace is used.

        Delete can occur in the following mutual exclusive ways:
        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
            (note that for this option delete all must be set to False)

        API reference: https://docs.pinecone.io/reference/delete_post

        Examples:
            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.delete(delete_all=True, namespace='my_namespace')
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace')

        Args:
            ids (List[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                                Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                            If not specified, the default namespace is used.
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Keyword Args:
          Supports OpenAPI client keyword arguments. See pinecone.core.client.models.DeleteRequest for more details.


          Returns: An empty dictionary if the delete operation was successful.
        """
        pass

    @abstractmethod
    async def fetch(
        self, ids: List[str], namespace: Optional[str] = None, **kwargs
    ) -> FetchResponse:
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        API reference: https://docs.pinecone.io/reference/fetch

        Examples:
            >>> index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.fetch(ids=['id1', 'id2'])

        Args:
            ids (List[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
        Keyword Args:
            Supports OpenAPI client keyword arguments. See pinecone.core.client.models.FetchResponse for more details.


        Returns: FetchResponse object which contains the list of Vector objects, and namespace name.
        """
        pass

    @abstractmethod
    async def query(
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
    ) -> QueryResponse:
        """
        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        API reference: https://docs.pinecone.io/reference/query

        Examples:
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
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (Dict[str, Union[str, float, int, bool, List, dict]):
                    The filter to apply. You can use vector metadata to limit your search.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
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
    async def query_namespaces(
        self,
        namespaces: List[str],
        top_k: Optional[int] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        vector: Optional[List[float]] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        """The query_namespaces() method is used to make a query to multiple namespaces in parallel and combine the results into one result set.

        Since several asynchronous calls are made on your behalf when calling this method, you will need to tune the pool_threads and connection_pool_maxsize parameter of the Index constructor to suite your workload.

        Examples:

        ```python
        from pinecone import Pinecone

        pc = Pinecone(api_key="your-api-key")
        index = pc.Index(
            host="index-name",
            pool_threads=32,
            connection_pool_maxsize=32
        )

        query_vec = [0.1, 0.2, 0.3] # An embedding that matches the index dimension
        combined_results = index.query_namespaces(
            vector=query_vec,
            namespaces=['ns1', 'ns2', 'ns3', 'ns4'],
            top_k=10,
            filter={'genre': {"$eq": "drama"}},
            include_values=True,
            include_metadata=True
        )
        for vec in combined_results.matches:
            print(vec.id, vec.score)
        print(combined_results.usage)
        ```

        Args:
            vector (List[float]): The query vector, must be the same length as the dimension of the index being queried.
            namespaces (List[str]): The list of namespaces to query.
            top_k (Optional[int], optional): The number of results you would like to request from each namespace. Defaults to 10.
            filter (Optional[Dict[str, Union[str, float, int, bool, List, dict]]], optional): Pass an optional filter to filter results based on metadata. Defaults to None.
            include_values (Optional[bool], optional): Boolean field indicating whether vector values should be included with results. Defaults to None.
            include_metadata (Optional[bool], optional): Boolean field indicating whether vector metadata should be included with results. Defaults to None.
            sparse_vector (Optional[ Union[SparseValues, Dict[str, Union[List[float], List[int]]]] ], optional): If you are working with a dotproduct index, you can pass a sparse vector as part of your hybrid search. Defaults to None.

        Returns:
            QueryNamespacesResults: A QueryNamespacesResults object containing the combined results from all namespaces, as well as the combined usage cost in read units.
        """
        pass

    @abstractmethod
    async def update(
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

        API reference: https://docs.pinecone.io/reference/update

        Examples:
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

        Keyword Args:
            Supports OpenAPI client keyword arguments. See pinecone.core.client.models.UpdateRequest for more details.

        Returns: An empty dictionary if the update was successful.
        """
        pass

    @abstractmethod
    async def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        API reference: https://docs.pinecone.io/reference/describe_index_stats_post

        Examples:
            >>> index.describe_index_stats()
            >>> index.describe_index_stats(filter={'key': 'value'})

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.
        """
        pass

    @abstractmethod
    async def list_paginated(
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
    async def list(self, **kwargs):
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
    async def upsert_records(self, namespace: str, records: List[Dict]):
        """
        Upsert records to a namespace.

        Converts records into embeddings and upserts them into a namespacce in the index.

        :param namespace: The namespace of the index to upsert records to.
        :type namespace: str, required
        :param records: The records to upsert into the index.
        :type records: List[Dict], required
        """
        pass

    @abstractmethod
    async def search(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """
        Search for records.

        This operation converts a query to a vector embedding and then searches a namespace. You
        can optionally provide a reranking operation as part of the search.

        :param namespace: The namespace in the index to search.
        :type namespace: str, required
        :param query: The SearchQuery to use for the search.
        :type query: Union[Dict, SearchQuery], required
        :param rerank: The SearchRerank to use with the search request.
        :type rerank: Union[Dict, SearchRerank], optional
        :return: The records that match the search.
        :rtype: RecordModel
        """
        pass

    @abstractmethod
    async def search_records(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """Alias of the search() method."""
        pass
