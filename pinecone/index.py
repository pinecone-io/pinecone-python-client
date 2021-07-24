from typing import NamedTuple, Iterable, Tuple, Iterator, List
import numpy as np

from pinecone.protos import core_pb2
from pinecone.utils import load_numpy, load_strings
from pinecone.utils.sentry import sentry_decorator as sentry
from .api_controller import ControllerAPI
from .constants import Config
from .grpc import GRPCClient, GRPCClientConfig, Cursor

__all__ = ["UpsertResult", "DeleteResult", "QueryResult", "FetchResult", "InfoResult", "Index"]


class UpsertResult(NamedTuple):
    """Result of an index request."""

    id: str


class DeleteResult(NamedTuple):
    """Result of a delete request."""

    id: str


class QueryResult(NamedTuple):
    """Result of a query request."""

    ids: Iterable[str]
    scores: Iterable[float]
    data: np.ndarray = None


class FetchResult(NamedTuple):
    """Result of a fetch request"""

    id: str
    vector: np.ndarray


class InfoResult(NamedTuple):
    """Metadata of an index."""

    index_size: str
    dimension: int


class ListResult(NamedTuple):
    """Metadata of an index."""

    items: List[str] = None


class Index:
    def __init__(self, name: str, **kwargs):
        """Pinecone Index.

        :param name: name of Pinecone index.
        :type name: str
        :param `**kwargs`:
            See below

        :Keyword Arguments:
            * **batch_size** (*int*) --
              the number of items to batch-process. Defaults to 100.
            * **disable_progress_bar** (*bool*) --
              whether to disable progress bar. Defaults to False.
            * **grpc_config** (*dict*) --
              See :class:`pinecone.grpc.GRPCClientConfig` for grpc client related configurations.
        """
        self.name = name
        self.batch_size = kwargs.pop("batch_size", 100)
        self.disable_progress_bar = kwargs.pop("disable_progress_bar", False)
        self.grpc = self._connect(kwargs.pop("grpc_config", {}))

    def _connect(self, grpc_config: dict):
        """Sets up a connection to an index."""
        api = ControllerAPI(host=Config.CONTROLLER_HOST, api_key=Config.API_KEY)
        status = api.get_status(self.name)
        if not status.get("ready"):
            raise ConnectionError

        if self.name not in api.list_services():
            raise RuntimeError("Index '{}' is not found.".format(self.name))

        host = status.get("host") or Config.CONTROLLER_HOST.split("://")[1].split(":")[0]
        port = status.get("port", -1)
        config = GRPCClientConfig._from_dict(grpc_config)
        return GRPCClient(host=host, port=port, api_key=Config.API_KEY, service_name=self.name, **(config._asdict()))

    def _make_cursor_config(self, **kwargs) -> dict:
        """Assembles cursor configurations."""
        config = {
            "grpc_client": self.grpc,
            "batch_size": self.batch_size,
            "disable_progress_bar": self.disable_progress_bar,
        }
        if "batch_size" in kwargs:
            config["batch_size"] = kwargs.pop("batch_size")
        if "disable_progress_bar" in kwargs:
            config["disable_progress_bar"] = kwargs.pop("disable_progress_bar")
        config.update(**kwargs)
        return config

    def _upsert(self, items: Iterable[Tuple[str, np.ndarray]], namespace: str = None, **kwargs) -> Cursor:
        def _create_index_request(item_batch: Iterable[Tuple[str, np.ndarray]]) -> "core_pb2.Request":
            """Creates an index request from the items."""
            path = "write"
            ids_buffer, vectors_buffer = list(zip(*item_batch))
            ids_buffer = list(map(str, ids_buffer))
            return self.grpc.get_index_request(ids=ids_buffer, data=vectors_buffer, namespace=namespace, path=path)

        def _parse_index_response(response) -> List[UpsertResult]:
            """Parses an index response"""
            return [UpsertResult(id=_id) for _id in response.index.ids]

        config = self._make_cursor_config(
            items=items, create_request_fn=_create_index_request, parse_response_fn=_parse_index_response, **kwargs
        )
        return Cursor(**config)

    def _query(
        self,
        queries: Iterable[np.ndarray],
        namespace: str = None,
        top_k: int = 10,
        include_data: bool = False,
        top_k_overrides: Iterable[int] = None,
        namespace_overrides: Iterable[str] = None,
        **kwargs
    ) -> Cursor:
        def _create_query_request(query_batch: Iterable[np.ndarray]) -> "core_pb2.Request":
            """Generates a query request."""
            path = "read"
            return self.grpc.get_query_request(
                data=query_batch, top_k=top_k, include_data=include_data, namespace=namespace, path=path,
                namespace_overrides=namespace_overrides, top_k_overrides=top_k_overrides
            )

        def _parse_query_response(response) -> List[QueryResult]:
            """Parses a query response"""
            return [
                QueryResult(
                    ids=load_strings(matches.ids),
                    scores=load_numpy(matches.scores),
                    data=load_numpy(matches.data) if response.query.include_data else None,
                )
                for matches in response.query.matches
            ]

        config = self._make_cursor_config(
            items=queries, create_request_fn=_create_query_request, parse_response_fn=_parse_query_response, **kwargs
        )
        return Cursor(**config)

    def _fetch(self, ids: Iterable[str], namespace: str = None, **kwargs) -> Cursor:
        def _create_fetch_request(id_batch: Iterable[str]) -> "core_pb2.Request":
            """Generates a delete request."""
            path = "read"
            return self.grpc.get_fetch_request(ids=id_batch, namespace=namespace, path=path)

        def _parse_fetch_response(response) -> List[FetchResult]:
            return [
                FetchResult(id=_id, vector=load_numpy(vector))
                for _id, vector in zip(response.fetch.ids, response.fetch.vectors)
            ]

        config = self._make_cursor_config(
            items=ids, create_request_fn=_create_fetch_request, parse_response_fn=_parse_fetch_response, **kwargs
        )
        return Cursor(**config)

    def _delete(self, ids: Iterable[str] = None, delete_all: bool = False, namespace: str = None, **kwargs) -> Cursor:
        def _create_delete_request(id_batch: Iterable[str]) -> "core_pb2.Request":
            """Generates a delete request."""
            path = "write"
            return self.grpc.get_delete_request(ids=id_batch, delete_all=delete_all, namespace=namespace, path=path)

        def _parse_delete_response(response) -> List[DeleteResult]:
            return [DeleteResult(id=_id) for _id in response.delete.ids]

        if delete_all:
            ids = ['#all']
        elif ids is None:
            ids = []

        config = self._make_cursor_config(
            items=ids, create_request_fn=_create_delete_request, parse_response_fn=_parse_delete_response, **kwargs
        )
        return Cursor(**config)

    @sentry
    def upsert(
        self, items: Iterable[Tuple[str, np.ndarray]], namespace: str = None, **kwargs
    ) -> Iterable[UpsertResult]:
        """Inserts or updates items.

        An item is an `(id, vector)` tuple.

        Insert an item into the index, or update the item's value by the item's id.

        :param items: tuples of the form (id, numpy.ndarray). The length of an id is limited to **64** characters.
        :type items: Iterable[Tuple[str, np.ndarray]]
        :param namespace: a partition in an index. Defaults to None.
        :type namespace: str, optional
        :param `**kwargs`:
            See below

        :Keyword Arguments:
            * **batch_size** (*int*) --
              overrides the number of items to batch-process.
            * **disable_progress_bar** (*bool*) --
              overrides whether to disable the progress bar

        :return: Acknowledgements of items indexed.
        :rtype: Iterable[UpsertResult]
        """
        return self._upsert(items=items, namespace=namespace, **kwargs).collect()

    @sentry
    def unary_upsert(self, item: Tuple[str, np.ndarray], namespace: str = None, **kwargs) -> UpsertResult:
        """Upserts one item at a time.

        This method sends a unary gRPC request.
        See :py:meth:`Index.upsert` for details about the parameters.
        """
        results = self._upsert(
            items=[item], namespace=namespace, unary=True, disable_progress_bar=True, **kwargs
        ).collect()
        return next(iter(results))

    @sentry
    def query(
        self,
        queries: Iterable[np.ndarray],
        namespace: str = None,
        top_k: int = 10,
        include_data: bool = False,
        **kwargs
    ) -> Iterable[QueryResult]:
        """Sends queries to the index and returns the top results ordered by their scores.

        :param queries: queries
        :type queries: Iterable[np.ndarray]
        :param namespace: a partition in an index. Use default namespace when not specified.
        :type namespace: str, optional
        :param top_k: defaults to 10, the number of top query results to return for each query,
            ordered by their scores
        :type top_k: int, optional
        :param `**kwargs`:
            See below

        :Keyword Arguments:
            * **batch_size** (*int*) --
              overrides the number of items to batch-process.
            * **disable_progress_bar** (*bool*) --
              overrides whether to disable the progress bar

        :return: For each query, retrieve its nearest neighbors in the index.
        :rtype: Iterable[QueryResult]
        """
        return self._query(
            queries=queries, namespace=namespace, top_k=top_k, include_data=include_data, **kwargs
        ).collect()

    @sentry
    def unary_query(
        self, query: np.ndarray, namespace: str = None, top_k: int = 10, include_data: bool = False, **kwargs
    ) -> UpsertResult:
        """Sends one query at a time.

        This method sends a unary gRPC request.
        See :py:meth:`Index.query` for details about the parameters.
        """
        results = self._query(
            queries=[query],
            namespace=namespace,
            top_k=top_k,
            include_data=include_data,
            unary=True,
            disable_progress_bar=True,
            **kwargs
        ).collect()
        return next(iter(results))

    @sentry
    def fetch(self, ids: Iterable[str], namespace: str = None, **kwargs) -> Iterable[FetchResult]:
        """Fetches items by their ids.

        :param ids: ids of items
        :type ids: Iterable[str]
        :param namespace: a partition in an index. Use default namespace when not specified.
        :type namespace: str, optional
        :param `**kwargs`:
            See below

        :Keyword Arguments:
            * **batch_size** (*int*) --
              overrides the number of items to batch-process.
            * **disable_progress_bar** (*bool*) --
              overrides whether to disable the progress bar

        :return: id-vector pairs stored in the index.
        :rtype: Iterable[FetchResult]
        """
        return self._fetch(ids=ids, namespace=namespace, **kwargs).collect()

    @sentry
    def unary_fetch(self, id: str, namespace: str = None, **kwargs) -> FetchResult:
        """Fetches one item at a time.

        This method sends a unary gRPC request.
        See :py:meth:`Index.fetch` for details about the parameters.
        """
        results = self._fetch(ids=[id], namespace=namespace, unary=True, disable_progress_bar=True, **kwargs).collect()
        return next(iter(results))

    @sentry
    def delete(self, ids: Iterable[str] = None, delete_all: bool = False, namespace: str = None, **kwargs) -> Iterable[DeleteResult]:
        """Deletes items by their ids.

        :param ids: ids of items
        :type ids: Iterable[str], optional
        :param delete_all: if True, delete all ids in the index/namespace. defaults to False
        :type delete_all: bool
        :param namespace: a partition in an index. Use default namespace when not specified.
        :type namespace: str, optional
        :param `**kwargs`:
            See below

        :Keyword Arguments:
            * **batch_size** (*int*) --
              overrides the number of items to batch-process.
            * **disable_progress_bar** (*bool*) --
              overrides whether to disable the progress bar

        :return: Acknowledgements of items deleted from the index.
        :rtype: Iterable[DeleteResult]
        """
        return self._delete(ids=ids, delete_all=delete_all, namespace=namespace, **kwargs).collect()

    @sentry
    def unary_delete(self, id: str, namespace: str = None, **kwargs) -> DeleteResult:
        """Deletes one item at a time.

        This method sends a unary gRPC request.
        See :py:meth:`Index.delete` for details about the parameters.
        """
        results = self._delete(ids=[id], namespace=namespace, unary=True, disable_progress_bar=True, **kwargs).collect()
        return next(iter(results))

    @sentry
    def info(self, namespace: str = None) -> InfoResult:
        """Returns information about the index.
        
        :param namespace: a partition in an index. Use default namespace when not specified.
        :type namespace: str, optional

        :return: Information about the index.
        :rtype: InfoResult
        """

        def _create_info_request(*args) -> "core_pb2.Request":
            path = "write"
            return self.grpc.get_info_request(namespace=namespace, path=path)

        def _parse_info_response(response) -> List[InfoResult]:
            return [InfoResult(index_size=response.info.index_size, dimension=response.info.dimension)]

        config = self._make_cursor_config(
            items=[1],
            create_request_fn=_create_info_request,
            parse_response_fn=_parse_info_response,
            unary=True,
            disable_progress_bar=True,
        )
        return next(iter(Cursor(**config).collect()))

    @sentry
    def list(self, resource_type: str, namespace: str = None) -> ListResult:
        """Returns list of resources in the the index.

        :param resource_type: type of resources to be listed. For now, only 'ids' and 'namespaces' are supported
        :type resource_type: str
        :param namespace: a partition in an index. Use default namespace when not specified.
        :type namespace: str, optional

        :return: Information about the index.
        :rtype: InfoResult
        """

        def _create_list_request(*args) -> "core_pb2.Request":
            path = "write"
            return self.grpc.get_list_request(resource_type=resource_type, namespace=namespace, path=path)

        def _parse_list_response(response) -> List[ListResult]:
            return [ListResult(items=load_strings(response.list.items))]

        config = self._make_cursor_config(
            items=[1],
            create_request_fn=_create_list_request,
            parse_response_fn=_parse_list_response,
            unary=True,
            disable_progress_bar=True,
        )
        return next(iter(Cursor(**config).collect()))

    @sentry
    def close(self):
        """Closes the connection to the index."""

        self.grpc.channel.close()
