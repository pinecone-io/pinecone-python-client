import asyncio
from functools import wraps
from typing import Dict, Tuple, Optional

from grpc._channel import _InactiveRpcError

from pinecone import Config
from .utils import _generate_request_id
from .config import GRPCClientConfig
from pinecone.utils.constants import REQUEST_ID, CLIENT_VERSION
from grpc import CallCredentials, Compression, StatusCode
from grpc.aio import AioRpcError
from google.protobuf.message import Message

from pinecone.exceptions import (
    PineconeException,
    PineconeApiValueError,
    PineconeApiException,
    UnauthorizedException,
    PineconeNotFoundException,
    ServiceException,
)


class GrpcRunner:
    def __init__(self, index_name: str, config: Config, grpc_config: GRPCClientConfig):
        self.config = config
        self.grpc_client_config = grpc_config

        self.fixed_metadata = {
            "api-key": config.api_key,
            "service-name": index_name,
            "client-version": CLIENT_VERSION,
        }
        if self.grpc_client_config.additional_metadata:
            self.fixed_metadata.update(self.grpc_client_config.additional_metadata)

    def run(
        self,
        func,
        request: Message,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        credentials: Optional[CallCredentials] = None,
        wait_for_ready: Optional[bool] = None,
        compression: Optional[Compression] = None,
    ):
        @wraps(func)
        def wrapped():
            user_provided_metadata = metadata or {}
            _metadata = self._prepare_metadata(user_provided_metadata)
            try:
                return func(
                    request,
                    timeout=timeout,
                    metadata=_metadata,
                    credentials=credentials,
                    wait_for_ready=wait_for_ready,
                    compression=compression,
                )
            except _InactiveRpcError as e:
                self._map_exception(e, e._state.code, e._state.details)

        return wrapped()

    async def run_asyncio(
        self,
        func,
        request: Message,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        credentials: Optional[CallCredentials] = None,
        wait_for_ready: Optional[bool] = None,
        compression: Optional[Compression] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ):
        @wraps(func)
        async def wrapped():
            user_provided_metadata = metadata or {}
            _metadata = self._prepare_metadata(user_provided_metadata)
            try:
                if semaphore is not None:
                    async with semaphore:
                        return await func(
                            request,
                            timeout=timeout,
                            metadata=_metadata,
                            credentials=credentials,
                            wait_for_ready=wait_for_ready,
                            compression=compression,
                        )
                else:
                    return await func(
                        request,
                        timeout=timeout,
                        metadata=_metadata,
                        credentials=credentials,
                        wait_for_ready=wait_for_ready,
                        compression=compression,
                    )
            except AioRpcError as e:
                self._map_exception(e, e.code(), e.details())

        return await wrapped()

    def _prepare_metadata(
        self, user_provided_metadata: Dict[str, str]
    ) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (k, v)
            for k, v in {
                **self.fixed_metadata,
                **self._request_metadata(),
                **user_provided_metadata,
            }.items()
        )

    def _request_metadata(self) -> Dict[str, str]:
        return {REQUEST_ID: _generate_request_id()}

    def _map_exception(self, e: Exception, code: Optional[StatusCode], details: Optional[str]):
        # Client / connection issues
        details = details or ""

        if code in [StatusCode.DEADLINE_EXCEEDED]:
            raise TimeoutError(details) from e

        # Permissions stuff
        if code in [StatusCode.PERMISSION_DENIED, StatusCode.UNAUTHENTICATED]:
            raise UnauthorizedException(status=code, reason=details) from e

        # 400ish stuff
        if code in [StatusCode.NOT_FOUND]:
            raise PineconeNotFoundException(status=code, reason=details) from e
        if code in [StatusCode.INVALID_ARGUMENT, StatusCode.OUT_OF_RANGE]:
            raise PineconeApiValueError(details) from e
        if code in [
            StatusCode.ALREADY_EXISTS,
            StatusCode.FAILED_PRECONDITION,
            StatusCode.UNIMPLEMENTED,
            StatusCode.RESOURCE_EXHAUSTED,
        ]:
            raise PineconeApiException(status=code, reason=details) from e

        # 500ish stuff
        if code in [StatusCode.INTERNAL, StatusCode.UNAVAILABLE]:
            raise ServiceException(status=code, reason=details) from e
        if code in [StatusCode.UNKNOWN, StatusCode.DATA_LOSS, StatusCode.ABORTED]:
            # abandon hope, all ye who enter here
            raise PineconeException(code, details) from e

        # If you get here, you're in a bad place
        raise PineconeException(code, details) from e
