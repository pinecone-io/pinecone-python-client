#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from typing import Iterable, Callable, Iterator, Tuple, Optional, NamedTuple
import numpy as np

from pinecone.protos import core_pb2
from pinecone.utils import load_numpy, load_strings
from pinecone.utils.sentry import sentry_decorator as sentry
from pinecone.utils.progressbar import ProgressBar
from .api_controller import ControllerAPI
from .api_router import RouterAPI
from pinecone.constants import Config
from .grpc import GRPCClient

__all__ = ["connect", "Connection", "Cursor", "IndexResult", "DeleteResult", "QueryResult", "InfoResult", "FetchResult"]


class IndexResult(NamedTuple):
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


class InfoResult(NamedTuple):
    """Result of an info request"""

    index_size: str


class FetchResult(NamedTuple):
    """Result of a fetch request"""

    id: str
    vector: np.ndarray


class CursorConfig(NamedTuple):
    """Cursor configurations.


    :param grpc_client: the cursor's gRPC client.
    :type grpc_client: :class:`GRPCClient`
    :param create_request_fn: the function to create gRPC requests given the items.
    :type create_request_fn: Callable[[Iterable], "core_pb2.Request"]
    :param parse_response_fn: the function to parse gRPC response
    :type parse_response_fn: Optional[Callable[["core_pb2.Request"], Iterable]], optional
    :param unary: send unary-unary gRPC requests. Defaults to `False`.
    :type unary: bool, optional
    :param disable_progress_bar: disable progress bar. Defaults to `False`.
    :type disable_progress_bar: bool
    """

    grpc_client: GRPCClient
    create_request_fn: Callable[[Iterable], "core_pb2.Request"]
    parse_response_fn: Optional[Callable[["core_pb2.Request"], Iterable]] = None
    unary: Optional[bool] = False
    disable_progress_bar: bool = False


class Cursor:
    def __init__(
        self,
        config: CursorConfig,
        items: Iterable[np.ndarray],
        batch_size: Optional[int] = 100,
    ):
        """A cursor batch-processes requests sent to a service.

        :param config: cursor configurations
        :type config: :class:`CursorConfig`
        :param items: Items to process
        :type items: Iterable[np.ndarray]
        :param batch_size: the number of items to batch-process, defaults to 100
        :type batch_size: int, optional
        """
        self.batch_size = batch_size or 100
        self._item_iterator = iter(items)
        self._unary = config.unary
        self._disable_progress_bar = config.disable_progress_bar
        self._client = config.grpc_client
        self._create_reqeust_fn = config.create_request_fn
        self._parse_response_fn = config.parse_response_fn or self.identity

    @sentry
    def take(self, size: int):
        """Processes a given number of items and returns the results.

        :param size: the number of items to process
        """
        result = []
        if self._disable_progress_bar:
            result = list(self._fetch_unbuffered(size=size))
        else:
            result = list(ProgressBar.iter(self._fetch_unbuffered(size=size)))
        return result

    @sentry
    def collect(self):
        """Processes all of the items and returns the results."""
        result = []
        if self._disable_progress_bar:
            result = list(self._fetch_unbuffered())
        else:
            result = list(ProgressBar.iter(self._fetch_unbuffered()))
        return result

    @sentry
    def stream(self) -> Iterator:
        """Processes all of the items, returns the results one batch at a time."""
        ret_iter = None
        if self._disable_progress_bar:
            ret_iter = self._fetch_unbuffered()
        else:
            ret_iter = ProgressBar.iter(self._fetch_unbuffered())
        return ret_iter

    def __iter__(self):
        raise RuntimeError(
            "To iterate through the results, "
            "consider using one of the followoing methods: `take()`, `collect()`, `stream()`."
        )

    @classmethod
    def identity(cls, entity):
        """An identity function."""
        return entity

    def _fetch_unbuffered(self, size: int = None) -> Iterator:
        """Creates a generator to fetch results.

        :param size: how many items to fetch.
        """

        def parse_response(response):
            if response.status.code == core_pb2.Status.StatusCode.Value("ERROR"):
                exception = bool(response.status.details) and response.status.details[0].exception
                error_msg = exception or response.status.description or "Unknown error when fetching results."
                raise RuntimeError(error_msg)

            for elem in self._parse_response_fn(response):
                yield elem

        request_generator = self._generate_requests(size)
        if self._unary:
            for request in request_generator:
                response_ = self._client.send_request(request)
                for elem in parse_response(response_):
                    yield elem
        else:
            for response_ in self._client.stream_requests(request_generator):
                for elem in parse_response(response_):
                    yield elem

    def _generate_requests(self, size: int = None) -> Iterator["core_pb2.Request"]:
        """Generates protobuf requests.

        This method consumes the item iterator in batches,
        then creates requests that each wraps the batch of items.

        :param size: how many items to consume. If `None`, th method exhausts the item iterator.
        """
        batch = []
        for ii, item in enumerate(self._item_iterator):
            batch.append(item)
            if len(batch) >= self.batch_size or (size and len(batch) >= size):
                yield self._create_reqeust_fn(batch)
                batch = []
            if not size:
                # consume everything
                continue
            elif ii >= size - 1:
                break
        if batch:
            yield self._create_reqeust_fn(batch)


class Connection:
    """A connection manages communication with a service.

    You can only interact with a service (e.g. upsert, query)
    after a connection to the service has been established.

    Namespaces partition the items in an index. When you read from or write to a namespace in an index, you will
    only access items in that particular namespace.
    For instance, two namespaces can contain the items with the same ids but different values.
    Use namespaces when you want to use the same preprocessors and postprocessors for separate datasets.
    For example, if you are building a movie recommender system, then you could use namespaces to
    separate the recommendations by genre.
    """

    def __init__(
        self,
        host: str,
        port: int,
        api_key: str = None,
        service_name: str = None,
        timeout: int = 0,
        response_timeout: int = 20,
        traceroute: bool = False,
    ):
        """
        :param host: service host
        :type host: str
        :param port: service port
        :type port: int
        :param api_key: user API key, defaults to None
        :type api_key: str, optional
        :param service_name: name of the service, defaults to None
        :type service_name: str, optional
        :param timeout: defaults to 0. Timeout to retry connection if gRPC is unavailable. 0 is no retry.
        :type timeout: int, optional
        :param response_timeout: defaults to 20 seconds. Fail if gateway doesn't receive response within timeout.
        :type response_timeout: int, optional
        :param traceroute: Whether to send receipts back to the gateway from each stage of the graph
        :type traceroute: bool, optional
        """

        self.host = host
        self.port = port
        self.api_key = api_key
        self.service_name = service_name
        self.grpc = GRPCClient(
            host=host,
            port=port,
            api_key=api_key,
            service_name=service_name,
            conn_timeout=timeout,
            timeout=response_timeout,
            traceroute=traceroute,
        )

    @sentry
    def upsert(
        self,
        items: Iterable[Tuple[str, np.ndarray]],
        namespace: str = None,
        batch_size: int = None,
        cursor_config: dict = None,
    ) -> Cursor:
        """Inserts or updates items.

        An item is an `(id, vector)` tuple.

        Insert an item into the index, or update by item id if the item's id already exists.

        :param items: tuples of the form (id, numpy.ndarray). Id's length is limited to 64 characters.
        :type items: Iterable[Tuple[str, np.ndarray]]
        :param namespace: a partition in an index. Use default namespace when not specified.
        :type namespace: str, optional
        :param batch_size: overrides default batch size
        :type batch_size: int, optional
        :param cursor_config: additional cursor configs. See :class:`CursorConfig`.
        :type cursor_config: dict, optional
        :return: :class:`Cursor`
        """

        def _create_index_request(item_batch: Iterable[Tuple[str, np.ndarray]]) -> "core_pb2.Request":
            """Creates an index request using all of the items."""
            path = "write"
            ids_buffer, vectors_buffer = list(zip(*item_batch))
            ids_buffer = list(map(str, ids_buffer))
            return self.grpc.get_index_request(ids=ids_buffer, data=vectors_buffer, namespace=namespace, path=path)

        def _parse_index_response(response):
            return [IndexResult(id=_id) for _id in response.index.ids]

        config = CursorConfig(
            grpc_client=self.grpc,
            create_request_fn=_create_index_request,
            parse_response_fn=_parse_index_response,
            **(cursor_config or {})
        )
        return Cursor(config=config, items=items, batch_size=batch_size)

    @sentry
    def unary_upsert(self, item: Tuple[str, np.ndarray], namespace: str = None) -> Optional[IndexResult]:
        """Inserts or updates an item.

        This is the unary version of :py:meth:`upsert`, which upserts one item at a time.
        """
        result = self.upsert(
            items=[item], namespace=namespace, cursor_config={"unary": True, "disable_progress_bar": True}
        ).collect()
        return result[0] if result else None

    @sentry
    def delete(
        self, ids: Iterable[str], namespace: str = None, batch_size: int = None, cursor_config: dict = None
    ) -> Cursor:
        """Deletes items by their ids.

        :param ids: ids of items
        :type ids: Iterable[str]
        :param namespace: a partition in an index. Use default namespace when not specified.
        :type namespace: str, optional
        :param batch_size: overrides default batch size
        :type batch_size: int, optional
        :param cursor_config: additional cursor configs. See :class:`CursorConfig`.
        :type cursor_config: dict, optional
        :return: :class:`Cursor`
        """

        def _create_delete_request(id_batch: Iterable[str]) -> "core_pb2.Request":
            """Generates a delete request."""
            path = "write"
            return self.grpc.get_delete_request(ids=id_batch, namespace=namespace, path=path)

        def _parse_delete_response(response):
            return [DeleteResult(id=_id) for _id in response.delete.ids]

        config = CursorConfig(
            grpc_client=self.grpc,
            create_request_fn=_create_delete_request,
            parse_response_fn=_parse_delete_response,
            **(cursor_config or {})
        )
        return Cursor(config=config, items=ids, batch_size=batch_size)

    @sentry
    def unary_delete(self, id: str, namespace: str = None) -> Optional[DeleteResult]:
        """Deletes an item by its id.

        This is the unary version of :py:meth:`delete`, which deletes one item at a time.
        """
        result = self.delete(
            ids=[id], namespace=namespace, cursor_config={"unary": True, "disable_progress_bar": True}
        ).collect()
        return result[0] if result else None

    @sentry
    def fetch(
        self, ids: Iterable[str], namespace: str = None, batch_size: int = None, cursor_config: dict = None
    ) -> Cursor:
        """Fetches items by their ids.

        :param ids: ids of items
        :type ids: Iterable[str]
        :param namespace: a partition in an index. Use default namespace when not specified.
        :type namespace: str, optional
        :param batch_size: overrides default batch size
        :type batch_size: int, optional
        :param cursor_config: additional cursor configs. See :class:`CursorConfig`.
        :type cursor_config: dict, optional
        :return: :class:`Cursor`
        """

        def _create_fetch_request(id_batch: Iterable[str]) -> "core_pb2.Request":
            """Generates a delete request."""
            path = "read"
            return self.grpc.get_fetch_request(ids=id_batch, namespace=namespace, path=path)

        def _parse_fetch_response(response):
            return [
                FetchResult(id=_id, vector=load_numpy(vector))
                for _id, vector in zip(response.fetch.ids, response.fetch.vectors)
            ]

        config = CursorConfig(
            grpc_client=self.grpc,
            create_request_fn=_create_fetch_request,
            parse_response_fn=_parse_fetch_response,
            **(cursor_config or {})
        )
        return Cursor(config=config, items=ids, batch_size=batch_size)

    @sentry
    def unary_fetch(self, id: str, namespace: str = None) -> Optional[FetchResult]:
        """Fetches an item by its id.

        This is the unary version of :py:meth:`fetch`, which fetches one item at a time.
        """
        result = self.fetch(
            ids=[id], namespace=namespace, cursor_config={"unary": True, "disable_progress_bar": True}
        ).collect()
        return result[0] if result else None

    @sentry
    def query(
        self,
        queries: Iterable[np.ndarray],
        namespace: str = None,
        top_k: int = 10,
        batch_size: int = None,
        include_data: bool = False,
        cursor_config: dict = None,
        top_k_overrides: Iterable[int] = None,
        namespace_overrides: Iterable[str] = None
    ) -> Cursor:
        """Sends queries to the index and returns the top results ordered by their scores.

        :param queries: queries
        :type queries: Iterable[np.ndarray]
        :param namespace: a partition in an index. Use default namespace when not specified.
        :type namespace: str, optional
        :param top_k: defaults to 10, the number of top query results to return for each query,
            ordered by their scores
        :type top_k: int, optional
        :param batch_size: overrides default batch size
        :type batch_size: int, optional
        :param include_data: whether to return data associated with the query results, defaults to False
        :type include_data: bool, optional
        :param cursor_config: additional cursor configs. See :class:`CursorConfig`.
        :type cursor_config: dict, optional
        :return: :class:`Cursor`
        """

        def _create_query_request(query_batch: Iterable[np.ndarray]) -> "core_pb2.Request":
            """Generates a query request."""
            path = "read"
            return self.grpc.get_query_request(
                data=query_batch, top_k=top_k, include_data=include_data, namespace=namespace, path=path,
                namespace_overrides=namespace_overrides, top_k_overrides=top_k_overrides
            )

        def _parse_query_response(response):
            """Parses a query response"""
            return [
                QueryResult(
                    ids=load_strings(matches.ids),
                    scores=load_numpy(matches.scores),
                    data=load_numpy(matches.data) if response.query.include_data else None,
                )
                for matches in response.query.matches
            ]

        config = CursorConfig(
            grpc_client=self.grpc,
            create_request_fn=_create_query_request,
            parse_response_fn=_parse_query_response,
            **(cursor_config or {})
        )
        return Cursor(config=config, items=queries, batch_size=batch_size)

    @sentry
    def unary_query(
        self, query: np.ndarray, namespace: str = None, top_k: int = 10, include_data: bool = False
    ) -> Optional[QueryResult]:
        """Sends a query.

        This is the unary version of :py:meth:`query`, which sends one query at a time.
        """
        result = self.query(
            queries=[query],
            namespace=namespace,
            top_k=top_k,
            include_data=include_data,
            cursor_config={"unary": True, "disable_progress_bar": True},
        ).collect()
        return result[0] if result else None

    @sentry
    def info(self, namespace: str = None) -> Optional[InfoResult]:
        """Returns information of the index."""

        def _create_info_request(*args) -> "core_pb2.Request":
            path = "write"
            return self.grpc.get_info_request(namespace=namespace, path=path)

        def _parse_info_response(response):
            return [InfoResult(index_size=response.info.index_size)]

        cursor_config = CursorConfig(
            grpc_client=self.grpc,
            create_request_fn=_create_info_request,
            parse_response_fn=_parse_info_response,
            unary=True,
            disable_progress_bar=True,
        )
        result = Cursor(config=cursor_config, items=[1]).collect()
        return result[0] if result else None

    @sentry
    def close(self):
        """Closes the connection."""

        self.grpc.channel.close()

    def _set_secure(self, is_secure: bool):
        """Changes whether the grpc client is secure or not."""

        self.grpc.secure = is_secure


@sentry
def connect(
    service_name: str = None,
    router_name: str = None,
    timeout: int = 0,
    response_timeout: int = 20,
    traceroute: bool = False,
) -> Connection:
    """Connects to a live service/router and returns an instance of :class:`Connection`.

    :param service_name: name of a service.
    :param router_name: name of a router.
    :type service_name: str
    :type router_name: str
    :param timeout: gRPC connection timeout, defaults to 0
    :type timeout: int, optional
    :param timeout: defaults to 0. Timeout to retry connection if gRPC is unavailable. 0 is no retry.
    :type timeout: int, optional
    :param response_timeout: defaults to 20 seconds. Fail if gateway doesn't receive response within timeout.
    :type response_timeout: int, optional
    :param traceroute: Whether to send receipts back to the gateway from each stage of the graph
    :type traceroute: bool, optional
    :return: :class:`Connection`
    """
    dest_name = router_name or service_name
    if not dest_name:
        raise ValueError("Must supply service_name or router_name for connect()")
    api_cls = RouterAPI if router_name else ControllerAPI
    api = api_cls(host=Config.CONTROLLER_HOST, api_key=Config.API_KEY)
    status = api.get_status(dest_name)
    if not status.get("ready"):
        raise ConnectionError

    host = status.get("host") or Config.CONTROLLER_HOST.split("://")[1].split(":")[0]
    port = status.get("port")
    return Connection(
        host=host,
        port=port,
        api_key=Config.API_KEY,
        service_name=dest_name,
        timeout=timeout,
        response_timeout=response_timeout,
        traceroute=traceroute,
    )
