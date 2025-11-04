import logging
from typing import Optional, Dict, Union, List, Tuple, Any, Iterable, cast, Literal

from google.protobuf import json_format

from pinecone.utils.tqdm import tqdm
from pinecone.utils import require_kwargs
from concurrent.futures import as_completed, Future


from .utils import (
    dict_to_proto_struct,
    parse_fetch_response,
    parse_fetch_by_metadata_response,
    parse_query_response,
    parse_stats_response,
    parse_upsert_response,
    parse_update_response,
    parse_delete_response,
    parse_namespace_description,
    parse_list_namespaces_response,
)
from .vector_factory_grpc import VectorFactoryGRPC
from .sparse_values_factory import SparseValuesFactory

from pinecone.core.openapi.db_data.models import (
    FetchResponse,
    QueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
    NamespaceDescription,
    ListNamespacesResponse,
)
from pinecone.db_data.dataclasses import FetchByMetadataResponse
from pinecone.db_control.models.list_response import ListResponse as SimpleListResponse, Pagination
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
    Vector as GRPCVector,
    QueryVector as GRPCQueryVector,
    UpsertRequest,
    UpsertResponse,
    DeleteRequest,
    QueryRequest,
    FetchRequest,
    FetchByMetadataRequest,
    UpdateRequest,
    ListRequest,
    DescribeIndexStatsRequest,
    DeleteResponse,
    UpdateResponse,
    SparseValues as GRPCSparseValues,
    DescribeNamespaceRequest,
    DeleteNamespaceRequest,
    ListNamespacesRequest,
    CreateNamespaceRequest,
    MetadataSchema,
    MetadataFieldProperties,
)
from pinecone.core.grpc.protos.db_data_2025_10_pb2_grpc import VectorServiceStub
from pinecone import Vector, SparseValues
from pinecone.db_data.query_results_aggregator import QueryNamespacesResults, QueryResultsAggregator
from .base import GRPCIndexBase
from .future import PineconeGrpcFuture
from pinecone.db_data.types import (
    SparseVectorTypedDict,
    VectorTypedDict,
    VectorTuple,
    FilterTypedDict,
    VectorMetadataTypedDict,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .resources.vector import VectorResourceGRPC
    from .resources.namespace import NamespaceResourceGRPC


__all__ = [
    "GRPCIndex",
    "GRPCVector",
    "GRPCQueryVector",
    "GRPCSparseValues",
    "NamespaceDescription",
    "ListNamespacesResponse",
]

_logger = logging.getLogger(__name__)
""" :meta private: """


class GRPCIndex(GRPCIndexBase):
    """A client for interacting with a Pinecone index via GRPC API."""

    def __init__(
        self,
        index_name: str,
        config,
        channel=None,
        grpc_config=None,
        pool_threads=None,
        _endpoint_override=None,
    ):
        super().__init__(
            index_name=index_name,
            config=config,
            channel=channel,
            grpc_config=grpc_config,
            pool_threads=pool_threads,
            _endpoint_override=_endpoint_override,
        )
        self._vector_resource = None
        """ :meta private: """

        self._namespace_resource = None
        """ :meta private: """

    @property
    def stub_class(self):
        """:meta private:"""
        return VectorServiceStub

    @property
    def vector(self) -> "VectorResourceGRPC":
        """:meta private:"""
        if self._vector_resource is None:
            from .resources.vector import VectorResourceGRPC

            self._vector_resource = VectorResourceGRPC(
                stub=self.stub,
                runner=self.runner,
                threadpool_executor=self.threadpool_executor,
            )
        return self._vector_resource

    @property
    def namespace(self) -> "NamespaceResourceGRPC":
        """:meta private:"""
        if self._namespace_resource is None:
            from .resources.namespace import NamespaceResourceGRPC

            self._namespace_resource = NamespaceResourceGRPC(
                stub=self.stub,
                runner=self.runner,
            )
        return self._namespace_resource

    def upsert(
        self,
        vectors: Union[List[Vector], List[GRPCVector], List[VectorTuple], List[VectorTypedDict]],
        async_req: bool = False,
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Union[UpsertResponse, PineconeGrpcFuture]:
        """
        The upsert operation writes vectors into a namespace.
        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        Examples:

        .. code-block:: python

            >>> index.upsert([('id1', [1.0, 2.0, 3.0], {'key': 'value'}),
                              ('id2', [1.0, 2.0, 3.0])
                              ],
                              namespace='ns1', async_req=True)
            >>> index.upsert([{'id': 'id1', 'values': [1.0, 2.0, 3.0], 'metadata': {'key': 'value'}},
                              {'id': 'id2',
                                        'values': [1.0, 2.0, 3.0],
                                        'sparse_values': {'indices': [1, 8], 'values': [0.2, 0.4]},
                              ])
            >>> index.upsert([GRPCVector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
                              GRPCVector(id='id2', values=[1.0, 2.0, 3.0]),
                              GRPCVector(id='id3',
                                         values=[1.0, 2.0, 3.0],
                                         sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]))])

        Args:
            vectors (Union[List[Vector], List[Tuple]]): A list of vectors to upsert.

                     A vector can be represented by a 1) GRPCVector object, a 2) tuple or 3) a dictionary
                     1) if a tuple is used, it must be of the form (id, values, metadata) or (id, values).
                        where id is a string, vector is a list of floats, and metadata is a dict.
                        Examples: ('id1', [1.0, 2.0, 3.0], {'key': 'value'}), ('id2', [1.0, 2.0, 3.0])

                    2) if a GRPCVector object is used, a GRPCVector object must be of the form
                        GRPCVector(id, values, metadata), where metadata is an optional argument of type
                        Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]
                       Examples: GRPCVector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
                                 GRPCVector(id='id2', values=[1.0, 2.0, 3.0]),
                                 GRPCVector(id='id3',
                                            values=[1.0, 2.0, 3.0],
                                            sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]))

                    3) if a dictionary is used, it must be in the form
                       {'id': str, 'values': List[float], 'sparse_values': {'indices': List[int], 'values': List[float]},
                        'metadata': dict}

                    Note: the dimension of each vector must match the dimension of the index.
            async_req (bool): If True, the upsert operation will be performed asynchronously.
                              Cannot be used with batch_size.
                              Defaults to False. See: https://docs.pinecone.io/docs/performance-tuning [optional]
            namespace (str): The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size (int): The number of vectors to upsert in each batch.
                                Cannot be used with async_req=True.
                               If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress (bool): Whether to show a progress bar using tqdm.
                                  Applied only if batch_size is provided. Default is True.

        Returns: UpsertResponse, contains the number of vectors upserted
        """
        return self.vector.upsert(
            vectors=vectors,
            async_req=async_req,
            namespace=namespace,
            batch_size=batch_size,
            show_progress=show_progress,
            **kwargs,
        )

    def upsert_from_dataframe(
        self,
        df,
        namespace: str = "",
        batch_size: int = 500,
        use_async_requests: bool = True,
        show_progress: bool = True,
    ) -> UpsertResponse:
        """Upserts a dataframe into the index.

        Args:
            df: A pandas dataframe with the following columns: id, values, sparse_values, and metadata.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
            use_async_requests: Whether to upsert multiple requests at the same time using asynchronous request mechanism.
                                Set to ``False``
            show_progress: Whether to show a progress bar.
        """
        return self.vector.upsert_from_dataframe(
            df=df,
            namespace=namespace,
            batch_size=batch_size,
            use_async_requests=use_async_requests,
            show_progress=show_progress,
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        async_req: bool = False,
        **kwargs,
    ) -> Union[DeleteResponse, PineconeGrpcFuture]:
        """
        The Delete operation deletes vectors from the index, from a single namespace.
        No error raised if the vector id does not exist.

        Args:
            ids (List[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                               Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                             If not specified, the default namespace is used.
            filter (FilterTypedDict):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                     See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]
            async_req (bool): If True, the delete operation will be performed asynchronously.
                              Defaults to False. [optional]

        Returns: DeleteResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.

        .. admonition:: Note

            For any delete call, if namespace is not specified, the default namespace is used.

        Delete can occur in the following mutual exclusive ways:

        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
           (note that for this option delete all must be set to False)

        Examples:

        .. code-block:: python

            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.delete(delete_all=True, namespace='my_namespace')
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace', async_req=True)
        """
        return self.vector.delete(
            ids=ids,
            delete_all=delete_all,
            namespace=namespace,
            filter=filter,
            async_req=async_req,
            **kwargs,
        )

    def fetch(
        self,
        ids: Optional[List[str]],
        namespace: Optional[str] = None,
        async_req: Optional[bool] = False,
        **kwargs,
    ) -> Union[FetchResponse, PineconeGrpcFuture]:
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
        return self.vector.fetch(ids=ids, namespace=namespace, async_req=async_req, **kwargs)

    def fetch_by_metadata(
        self,
        filter: FilterTypedDict,
        namespace: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        async_req: Optional[bool] = False,
        **kwargs,
    ) -> Union[FetchByMetadataResponse, PineconeGrpcFuture]:
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
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                Metadata filter expression to select vectors.
                See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`
            namespace (str): The namespace to fetch vectors from.
                            If not specified, the default namespace is used. [optional]
            limit (int): Max number of vectors to return. Defaults to 100. [optional]
            pagination_token (str): Pagination token to continue a previous listing operation. [optional]
            async_req (bool): If True, the fetch operation will be performed asynchronously.
                             Defaults to False. [optional]

        Returns:
            FetchByMetadataResponse: Object containing the fetched vectors, namespace, usage, and pagination token.
        """
        return self.vector.fetch_by_metadata(
            filter=filter,
            namespace=namespace,
            limit=limit,
            pagination_token=pagination_token,
            async_req=async_req,
            **kwargs,
        )

    def query(
        self,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        top_k: Optional[int] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[
            Union[SparseValues, GRPCSparseValues, SparseVectorTypedDict]
        ] = None,
        async_req: Optional[bool] = False,
        **kwargs,
    ) -> Union[QueryResponse, PineconeGrpcFuture]:
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
            >>> index.query(vector=[1, 2, 3], sparse_vector=GRPCSparseValues([1, 2], [0.2, 0.4]),
            >>>             top_k=10, namespace='my_namespace')

        Args:
            vector (List[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each ``query()`` request can contain only one of the parameters
                                  ``id`` or ``vector``.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each ``query()`` request can contain only one of the parameters
                      ``vector`` or ``id``.. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
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
        return self.vector.query(
            vector=vector,
            id=id,
            namespace=namespace,
            top_k=top_k,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
            sparse_vector=sparse_vector,
            async_req=async_req,
            **kwargs,
        )

    def query_namespaces(
        self,
        vector: List[float],
        namespaces: List[str],
        metric: Literal["cosine", "euclidean", "dotproduct"],
        top_k: Optional[int] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        return self.vector.query_namespaces(
            vector=vector,
            namespaces=namespaces,
            metric=metric,
            top_k=top_k,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
            sparse_vector=sparse_vector,
            **kwargs,
        )

    def update(
        self,
        id: str,
        async_req: bool = False,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Union[UpdateResponse, PineconeGrpcFuture]:
        """
        The Update operation updates vector in a namespace.
        If a value is included, it will overwrite the previous value.
        If a set_metadata is included,
        the values of the fields specified in it will be added or overwrite the previous value.

        Examples:

        .. code-block:: python

            >>> index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace', async_req=True)
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>              namespace='my_namespace')
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]),
            >>>              namespace='my_namespace')

        Args:
            id (str): Vector's unique id.
            async_req (bool): If True, the update operation will be performed asynchronously.
                              Defaults to False. [optional]
            values (List[float]): vector values to set. [optional]
            set_metadata (Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]]):
                metadata to set for vector. [optional]
            namespace (str): Namespace name where to update the vector.. [optional]
            sparse_values: (Dict[str, Union[List[float], List[int]]]): sparse values to update for the vector.
                           Expected to be either a GRPCSparseValues object or a dict of the form:
                           {'indices': List[int], 'values': List[float]} where the lists each have the same length.


        Returns: UpdateResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.
        """
        return self.vector.update(
            id=id,
            async_req=async_req,
            values=values,
            set_metadata=set_metadata,
            namespace=namespace,
            sparse_values=sparse_values,
            **kwargs,
        )

    def list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> SimpleListResponse:
        """
        The list_paginated operation finds vectors based on an id prefix within a single namespace.
        It returns matching ids in a paginated form, with a pagination token to fetch the next page of results.
        This id list can then be passed to fetch or delete operations, depending on your use case.

        Consider using the ``list`` method to avoid having to handle pagination tokens manually.

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

        Returns: SimpleListResponse object which contains the list of ids, the namespace name, pagination information, and usage showing the number of read_units consumed.
        """
        return self.vector.list_paginated(
            prefix=prefix,
            limit=limit,
            pagination_token=pagination_token,
            namespace=namespace,
            **kwargs,
        )

    def list(self, **kwargs):
        """
        The list operation accepts all of the same arguments as list_paginated, and returns a generator that yields
        a list of the matching vector ids in each page of results. It automatically handles pagination tokens on your
        behalf.

        Examples:

        .. code-block:: python

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
        return self.vector.list(**kwargs)

    def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        Examples:

        .. code-block:: python

            >>> index.describe_index_stats()
            >>> index.describe_index_stats(filter={'key': 'value'})

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.
        """
        return self.vector.describe_index_stats(filter=filter, **kwargs)

    @require_kwargs
    def create_namespace(
        self, name: str, schema: Optional[Dict[str, Any]] = None, async_req: bool = False, **kwargs
    ) -> Union[NamespaceDescription, PineconeGrpcFuture]:
        """
        The create_namespace operation creates a namespace in a serverless index.

        Examples:

        .. code-block:: python

            >>> index.create_namespace(name='my_namespace')

            >>> # Create namespace asynchronously
            >>> future = index.create_namespace(name='my_namespace', async_req=True)
            >>> namespace = future.result()

        Args:
            name (str): The name of the namespace to create.
            schema (Optional[Dict[str, Any]]): Optional schema configuration for the namespace as a dictionary. [optional]
            async_req (bool): If True, the create_namespace operation will be performed asynchronously. [optional]

        Returns: NamespaceDescription object which contains information about the created namespace, or a PineconeGrpcFuture object if async_req is True.
        """
        return self.namespace.create(name=name, schema=schema, async_req=async_req, **kwargs)

    @require_kwargs
    def describe_namespace(self, namespace: str, **kwargs) -> NamespaceDescription:
        """
        The describe_namespace operation returns information about a specific namespace,
        including the total number of vectors in the namespace.

        Examples:

        .. code-block:: python

            >>> index.describe_namespace(namespace='my_namespace')

        Args:
            namespace (str): The namespace to describe.

        Returns: NamespaceDescription object which contains information about the namespace.
        """
        return self.namespace.describe(namespace=namespace, **kwargs)

    @require_kwargs
    def delete_namespace(self, namespace: str, **kwargs) -> Dict[str, Any]:
        """
        The delete_namespace operation deletes a namespace from an index.
        This operation is irreversible and will permanently delete all data in the namespace.

        Examples:

        .. code-block:: python

            >>> index.delete_namespace(namespace='my_namespace')

        Args:
            namespace (str): The namespace to delete.

        Returns: Empty dictionary indicating successful deletion.
        """
        return self.namespace.delete(namespace=namespace, **kwargs)

    @require_kwargs
    def list_namespaces_paginated(
        self, limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> ListNamespacesResponse:
        """
        The list_namespaces_paginated operation returns a list of all namespaces in a serverless index.
        It returns namespaces in a paginated form, with a pagination token to fetch the next page of results.

        Examples:

        .. code-block:: python

            >>> results = index.list_namespaces_paginated(limit=10)
            >>> [ns.name for ns in results.namespaces]
            ['namespace1', 'namespace2', 'namespace3']
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> next_results = index.list_namespaces_paginated(limit=10, pagination_token=results.pagination.next)

        Args:
            limit (Optional[int]): The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]

        Returns: ListNamespacesResponse object which contains the list of namespaces and pagination information.
        """
        return self.namespace.list_paginated(limit=limit, pagination_token=pagination_token, **kwargs)

    @require_kwargs
    def list_namespaces(self, limit: Optional[int] = None, **kwargs):
        """
        The list_namespaces operation accepts all of the same arguments as list_namespaces_paginated, and returns a generator that yields
        each namespace. It automatically handles pagination tokens on your behalf.

        Args:
            limit (Optional[int]): The maximum number of namespaces to fetch in each network call. If unspecified, the server will use a default value. [optional]

        Returns:
            Returns a generator that yields each namespace. It automatically handles pagination tokens on your behalf so you can
            easily iterate over all results. The ``list_namespaces`` method accepts all of the same arguments as list_namespaces_paginated

        Examples:

        .. code-block:: python

            >>> for namespace in index.list_namespaces():
            >>>     print(namespace.name)
            namespace1
            namespace2
            namespace3

        You can convert the generator into a list by wrapping the generator in a call to the built-in ``list`` function:

        .. code-block:: python

            namespaces = list(index.list_namespaces())

        You should be cautious with this approach because it will fetch all namespaces at once, which could be a large number
        of network calls and a lot of memory to hold the results.
        """
        return self.namespace.list(limit=limit, **kwargs)

    @staticmethod
    def _parse_non_empty_args(args: List[Tuple[str, Any]]) -> Dict[str, Any]:
        return {arg_name: val for arg_name, val in args if val is not None}
