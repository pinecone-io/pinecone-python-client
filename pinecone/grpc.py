#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
from typing import Iterable, List, Union, NamedTuple, Iterator, Callable, Any, Optional, Tuple
import grpc
import uuid
import atexit
import random
import time
import abc
import numpy as np
from pinecone import logger

from pinecone import utils
from pinecone.utils.constants import MAX_MSG_SIZE
from pinecone.utils.progressbar import ProgressBar
from pinecone.constants import CLIENT_VERSION
from pinecone.protos import core_pb2_grpc, core_pb2

__all__ = ["GRPCClient"]


class SleepPolicy(abc.ABC):
    @abc.abstractmethod
    def sleep(self, try_i: int):
        """
        How long to sleep in milliseconds.
        :param try_i: the number of retry (starting from zero)
        """
        assert try_i >= 0


class ExponentialBackoff(SleepPolicy):
    def __init__(self, *, init_backoff_ms: int, max_backoff_ms: int, multiplier: int):
        self.init_backoff = random.randint(0, init_backoff_ms)
        self.max_backoff = max_backoff_ms
        self.multiplier = multiplier

    def sleep(self, try_i: int):
        sleep_range = min(self.init_backoff * self.multiplier ** try_i, self.max_backoff)
        sleep_ms = random.randint(0, sleep_range)
        logger.debug(f"gRPC retry. Sleeping for {sleep_ms}ms")
        time.sleep(sleep_ms / 1000)


class RetryOnRpcErrorClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """gRPC retry.

    Referece: https://github.com/grpc/grpc/issues/19514#issuecomment-531700657
    """

    def __init__(
        self, *, max_attempts: int, sleep_policy: SleepPolicy, retryable_status: Optional[Tuple[grpc.StatusCode]] = None
    ):
        self.max_attempts = max_attempts
        self.sleep_policy = sleep_policy
        self.retryable_status = retryable_status or ()

    def _is_retryable_error(self, response_or_error):
        """Determine if a response is a retryable error."""
        return (
            isinstance(response_or_error, grpc.RpcError)
            and "_MultiThreadedRendezvous" not in response_or_error.__class__.__name__
            and response_or_error.code() in self.retryable_status
        )

    def _intercept_call(self, continuation, client_call_details, request_or_iterator):
        response = None
        for try_i in range(self.max_attempts):
            response = continuation(client_call_details, request_or_iterator)
            if not self._is_retryable_error(response):
                break
            self.sleep_policy.sleep(try_i)
        return response

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return self._intercept_call(continuation, client_call_details, request_iterator)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return self._intercept_call(continuation, client_call_details, request_iterator)


class GRPCClientConfig(NamedTuple):
    """
    GRPC client configuration options.

    :param secure: Whether to use encrypted protocol (SSL). defaults to True.
    :type traceroute: bool, optional
    :param timeout: defaults to 2 seconds. Fail if gateway doesn't receive response within timeout.
    :type timeout: int, optional
    :param conn_timeout: defaults to 1. Timeout to retry connection if gRPC is unavailable. 0 is no retry.
    :type conn_timeout: int, optional
    :param traceroute: Whether to send receipts back to the gateway from each stage of the graph
    :type traceroute: bool, optional
    :param reuse_channel: Whether to reuse the same grpc channel for multiple requests
    :type traceroute: bool, optional
    """
    secure: bool = True
    timeout: int = 20
    conn_timeout: int = 1
    traceroute: bool = False
    reuse_channel: bool = True
    retry_config: Optional[dict] = None

    @classmethod
    def _from_dict(cls, kwargs: dict):
        cls_kwargs = {kk: vv for kk, vv in kwargs.items() if kk in cls._fields}
        return cls(**cls_kwargs)


class GRPCClient:
    """Client for gRPC."""

    def __init__(self, host: str = "", port: int = -1, service_name: str = "", api_key: str = "", **kwargs):
        self.host = host
        self.port = port
        self.metadata = (("api-key", api_key), ("service-name", service_name))
        config = GRPCClientConfig._from_dict(kwargs)
        self.secure = config.secure
        self.timeout = config.timeout
        self.traceroute = config.traceroute
        self.conn_timeout = config.conn_timeout
        self.reuse_channel = config.reuse_channel
        self.retry_config = config.retry_config or {
            "max_attempts": 4,
            "sleep_policy": ExponentialBackoff(init_backoff_ms=100, max_backoff_ms=1600, multiplier=2),
            "retryable_status": (grpc.StatusCode.UNAVAILABLE,),
        }
        self._channel = None
        atexit.register(_clean_up_client, client=self)

    @property
    def channel(self):
        """Creates GRPC channel."""

        def _gen():
            target = "{}:{}".format(self.host, self.port)
            options = (
                ("grpc.max_send_message_length", MAX_MSG_SIZE),
                ("grpc.max_receive_message_length", MAX_MSG_SIZE),
            )
            channel = None
            if not self.secure:
                channel = grpc.insecure_channel(target, options=options)
            else:
                tls = grpc.ssl_channel_credentials()
                channel = grpc.secure_channel(
                    target, tls, options=(("grpc.ssl_target_name_override", self.host),) + options
                )
            interceptor = RetryOnRpcErrorClientInterceptor(**self.retry_config)
            return grpc.intercept_channel(channel, interceptor)

        if self.reuse_channel and self._channel and self.grpc_server_on():
            return self._channel
        self._channel = _gen()
        return self._channel

    def get_index_request(
        self,
        ids: List[str] = None,
        data: Union[np.ndarray, List] = None,
        path: str = None,
        namespace: str = None,
    ) -> "core_pb2.Request":
        """Returns an upsert request."""
        req = core_pb2.IndexRequest()
        if ids is not None:
            req.ids[:] = [str(ee) for ee in ids]
        if data is not None:
            req.data.CopyFrom(utils.dump_numpy(_to_ndarray(data)))
        return core_pb2.Request(
            request_id=_generate_request_id(),
            version=CLIENT_VERSION,
            index=req,
            path=path,
            namespace=namespace,
            timeout=self.timeout,
            traceroute=self.traceroute,
        )

    def get_delete_request(
        self,
        ids: Iterable[str],
        delete_all: bool = False,
        path: str = None,
        namespace: str = None,
    ) -> "core_pb2.Request":
        """Returns a delete request."""
        req = core_pb2.DeleteRequest(ids=ids, delete_all=delete_all)
        return core_pb2.Request(
            request_id=_generate_request_id(),
            version=CLIENT_VERSION,
            delete=req,
            path=path,
            namespace=namespace,
            timeout=self.timeout,
            traceroute=self.traceroute,
        )

    def get_fetch_request(
        self,
        ids: Iterable[str],
        path: str = None,
        namespace: str = None,
    ) -> "core_pb2.Request":
        """Returns a fetch request."""
        req = core_pb2.FetchRequest(ids=ids)
        return core_pb2.Request(
            request_id=_generate_request_id(),
            version=CLIENT_VERSION,
            fetch=req,
            path=path,
            namespace=namespace,
            timeout=self.timeout,
            traceroute=self.traceroute,
        )

    def get_query_request(
            self,
            top_k: int = None,
            include_data: bool = None,
            data: Union[np.ndarray, List] = None,
            matches: List[dict] = None,
            path: str = None,
            namespace: str = None,
            namespace_overrides: List[str] = None,
            top_k_overrides: List[int] = None
    ) -> "core_pb2.Request":
        """Returns a query request."""
        req = core_pb2.QueryRequest()
        if top_k is not None:
            req.top_k = top_k
        if include_data is not None:
            req.include_data = include_data
        if data is not None:
            req.data.CopyFrom(utils.dump_numpy(_to_ndarray(data)))
        if top_k_overrides:
            req.top_k_overrides.extend(top_k_overrides)
        if namespace_overrides:
            req.namespace_overrides.extend(namespace_overrides)
        if matches is not None:
            req.matches.extend(
                [
                    core_pb2.ScoredResults(
                        ids=utils.dump_strings(mat.get("ids")),
                        scores=utils.dump_numpy(mat.get(np.array("scores"))),
                        data=utils.dump_numpy(np.array(mat["data"])) if mat.get("data") else None,
                    )
                    for mat in matches
                ]
            )
        return core_pb2.Request(
            request_id=_generate_request_id(),
            version=CLIENT_VERSION,
            query=req,
            path=path,
            namespace=namespace,
            timeout=self.timeout,
            traceroute=self.traceroute
        )

    def get_info_request(
        self,
        path: str = None,
        namespace: str = None,
    ) -> "core_pb2.Request":
        """Returns an info request"""
        return core_pb2.Request(
            request_id=_generate_request_id(),
            version=CLIENT_VERSION,
            info=core_pb2.InfoRequest(),
            path=path,
            namespace=namespace,
            timeout=self.timeout,
            traceroute=self.traceroute,
        )

    def get_list_request(
            self,
            resource_type: str,
            path: str = None,
            namespace: str = None,
    ) -> "core_pb2.Request":
        """Returns an info request"""
        return core_pb2.Request(
            request_id=_generate_request_id(),
            version=CLIENT_VERSION,
            list=core_pb2.ListRequest(resource_type=resource_type),
            path=path,
            namespace=namespace,
            timeout=self.timeout,
            traceroute=self.traceroute,
        )

    def grpc_server_on(self) -> bool:
        try:
            grpc.channel_ready_future(self._channel).result(timeout=self.conn_timeout)
            return True
        except grpc.FutureTimeoutError:
            return False

    def stream_requests(self, request_stream):
        """Creates sub to send proptobuf requests."""
        stub = core_pb2_grpc.RPCClientStub(self.channel)
        if self.conn_timeout > 0 and not self.grpc_server_on():
            raise ConnectionError("Failed to connect to gRPC host {}:{}".format(self.host, self.port))
        try:
            for response in stub.Call(request_stream, metadata=self.metadata):
                yield response
        except grpc.RpcError as rpc_error:
            raise ConnectionError(
                "Failed to connect to gRPC remote {}:{}. Status = {}. Details = {}. Debug = {}".format(
                    self.host, self.port, rpc_error.code(), rpc_error.details(), rpc_error.debug_error_string()
                )
            )

    def send_request(self, request):
        stub = core_pb2_grpc.RPCClientStub(self.channel)
        if self.conn_timeout > 0 and not self.grpc_server_on():
            raise ConnectionError("Failed to connect to gRPC host {}:{}".format(self.host, self.port))
        try:
            return stub.CallUnary(request, metadata=self.metadata)
        except grpc.RpcError as rpc_error:
            raise ConnectionError(
                "Failed to connect to gRPC remote {}:{}. Status = {}. Details = {}. Debug = {}".format(
                    self.host, self.port, rpc_error.code(), rpc_error.details(), rpc_error.debug_error_string()
                )
            )


class Cursor:
    _DEFAULT_BATCH_SIZE = 100

    def __init__(
        self,
        items: Iterable[np.ndarray],
        grpc_client: GRPCClient,
        create_request_fn: Callable[[Iterable], "core_pb2.Request"],
        parse_response_fn: Callable[["core_pb2.Request"], Iterable] = None,
        unary: bool = False,
        batch_size: int = None,
        disable_progress_bar: bool = False,
    ):
        """A cursor batch-processes requests sent to a service.

        :param config: cursor configurations
        :type config: :class:`CursorConfig`
        :param items: Items to process
        :type items: Iterable[np.ndarray]
        :param batch_size: the number of items to batch-process, defaults to 100
        :type batch_size: int, optional
        """
        self.batch_size = batch_size or self._DEFAULT_BATCH_SIZE
        self._item_iterator = iter(items)
        self.unary = unary
        self.disable_progress_bar = disable_progress_bar
        self.client = grpc_client
        self.create_reqeust_fn = create_request_fn
        self.parse_response_fn = parse_response_fn or self.identity

    def collect(self) -> Iterable:
        """Processes all of the items and returns the results."""
        result = []
        if self.disable_progress_bar:
            result = list(self._fetch_unbuffered())
        else:
            result = list(ProgressBar.iter(self._fetch_unbuffered()))
        return result

    def stream(self) -> Iterator:
        """Processes all of the items, returns the results one batch at a time."""
        ret_iter = None
        if self.disable_progress_bar:
            ret_iter = self._fetch_unbuffered()
        else:
            ret_iter = ProgressBar.iter(self._fetch_unbuffered())
        return ret_iter

    def __iter__(self):
        raise RuntimeError(
            "To iterate through the results, " "consider using one of the followoing methods: `collect()`, `stream()`."
        )

    @classmethod
    def identity(cls, entity: Any):
        """An identity function."""
        return entity

    def _fetch_unbuffered(self) -> Iterator:
        """Creates a generator to fetch results."""
        if self.unary:
            for request in self._generate_requests():
                response_ = self.client.send_request(request)
                for elem in self._parse_response(response_):
                    yield elem
        else:
            for response_ in self.client.stream_requests(self._generate_requests()):
                for elem in self._parse_response(response_):
                    yield elem

    def _iter_by_batch(self) -> Iterable[np.ndarray]:
        """Iterate the items in batches."""
        batch = []
        for item in self._item_iterator:
            batch.append(item)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _generate_requests(self) -> Iterator["core_pb2.Request"]:
        """Generates protobuf requests in batches."""
        for batch in self._iter_by_batch():
            yield self.create_reqeust_fn(batch)

    def _parse_response(self, response: "core_pb2.Request") -> Iterable:
        if response.status.code == core_pb2.Status.StatusCode.Value("ERROR"):
            exception = bool(response.status.details) and response.status.details[0].exception
            error_msg = exception or response.status.description or "Unknown error when fetching results."
            raise RuntimeError(error_msg)

        return self.parse_response_fn(response)


def _to_ndarray(arr: Iterable) -> np.ndarray:
    """Convert an array to a numpy array by making best guesses."""
    if isinstance(arr, np.ndarray):
        return arr
    else:
        return np.asarray(arr)


def _generate_request_id() -> int:
    return uuid.uuid4().int & (1 << 64) - 1


def _clean_up_client(client: GRPCClient):
    """Cleans up client resources"""
    try:
        client.channel.close()
    except TypeError:
        pass
