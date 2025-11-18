from functools import wraps
from typing import Any

from grpc._channel import _InactiveRpcError

from pinecone import Config
from .utils import _generate_request_id
from .config import GRPCClientConfig
from pinecone.utils.constants import REQUEST_ID, CLIENT_VERSION
from pinecone.exceptions.exceptions import PineconeException
from grpc import CallCredentials, Compression
from google.protobuf.message import Message
from pinecone.openapi_support.api_version import API_VERSION


class GrpcRunner:
    def __init__(self, index_name: str, config: Config, grpc_config: GRPCClientConfig):
        self.config = config
        self.grpc_client_config = grpc_config

        self.fixed_metadata = {
            "api-key": config.api_key,
            "service-name": index_name,
            "client-version": CLIENT_VERSION,
            "x-pinecone-api-version": API_VERSION,
        }
        if self.grpc_client_config.additional_metadata:
            self.fixed_metadata.update(self.grpc_client_config.additional_metadata)

    def run(
        self,
        func,
        request: Message,
        timeout: int | None = None,
        metadata: dict[str, str] | None = None,
        credentials: CallCredentials | None = None,
        wait_for_ready: bool | None = None,
        compression: Compression | None = None,
    ) -> tuple[Any, dict[str, str] | None]:
        """Run a GRPC call and return response with initial metadata.

        Returns:
            Tuple of (response, initial_metadata_dict). initial_metadata_dict may be None.
        """

        @wraps(func)
        def wrapped() -> tuple[Any, dict[str, str] | None]:
            user_provided_metadata = metadata or {}
            _metadata = self._prepare_metadata(user_provided_metadata)
            try:
                # For unary calls, use with_call to get trailing metadata
                # Check if func supports with_call (it's a method descriptor)
                if hasattr(func, "with_call") and callable(getattr(func, "with_call", None)):
                    try:
                        result = func.with_call(
                            request,
                            timeout=timeout,
                            metadata=_metadata,
                            credentials=credentials,
                            wait_for_ready=wait_for_ready,
                            compression=compression,
                        )
                        # Check if result is a tuple (real gRPC call)
                        if isinstance(result, tuple) and len(result) == 2:
                            response, call = result
                            # Extract initial metadata (sent from server at start of call)
                            initial_metadata = call.initial_metadata()
                            initial_metadata_dict = (
                                {key: value for key, value in initial_metadata}
                                if initial_metadata
                                else None
                            )
                            return response, initial_metadata_dict
                        # If with_call doesn't return a tuple, it's likely a mock - fall through to call func directly
                    except (TypeError, ValueError):
                        # If with_call fails or doesn't return expected format, fall back
                        pass
                # Fallback: call func directly (for mocks or methods without with_call)
                response = func(
                    request,
                    timeout=timeout,
                    metadata=_metadata,
                    credentials=credentials,
                    wait_for_ready=wait_for_ready,
                    compression=compression,
                )
                return response, None
            except _InactiveRpcError as e:
                raise PineconeException(e._state.debug_error_string) from e

        return wrapped()

    async def run_asyncio(
        self,
        func,
        request: Message,
        timeout: int | None = None,
        metadata: dict[str, str] | None = None,
        credentials: CallCredentials | None = None,
        wait_for_ready: bool | None = None,
        compression: Compression | None = None,
    ) -> tuple[Any, dict[str, str] | None]:
        """Run an async GRPC call and return response with initial metadata.

        Returns:
            Tuple of (response, initial_metadata_dict). initial_metadata_dict may be None.
        """

        @wraps(func)
        async def wrapped() -> tuple[Any, dict[str, str] | None]:
            user_provided_metadata = metadata or {}
            _metadata = self._prepare_metadata(user_provided_metadata)
            try:
                # For async unary calls, use with_call to get trailing metadata
                if hasattr(func, "with_call") and callable(getattr(func, "with_call", None)):
                    try:
                        result = await func.with_call(
                            request,
                            timeout=timeout,
                            metadata=_metadata,
                            credentials=credentials,
                            wait_for_ready=wait_for_ready,
                            compression=compression,
                        )
                        # Check if result is a tuple (real gRPC call)
                        if isinstance(result, tuple) and len(result) == 2:
                            response, call = result
                            # Extract initial metadata (sent from server at start of call)
                            initial_metadata = await call.initial_metadata()
                            initial_metadata_dict = (
                                {key: value for key, value in initial_metadata}
                                if initial_metadata
                                else None
                            )
                            return response, initial_metadata_dict
                        # If with_call doesn't return a tuple, it's likely a mock - fall through to call func directly
                    except (TypeError, ValueError):
                        # If with_call fails or doesn't return expected format, fall back
                        pass
                # Fallback: call func directly (for mocks or methods without with_call)
                response = await func(
                    request,
                    timeout=timeout,
                    metadata=_metadata,
                    credentials=credentials,
                    wait_for_ready=wait_for_ready,
                    compression=compression,
                )
                return response, None
            except _InactiveRpcError as e:
                raise PineconeException(e._state.debug_error_string) from e

        return await wrapped()

    def _prepare_metadata(
        self, user_provided_metadata: dict[str, str]
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (k, v)
            for k, v in {
                **self.fixed_metadata,
                **self._request_metadata(),
                **user_provided_metadata,
            }.items()
        )

    def _request_metadata(self) -> dict[str, str]:
        return {REQUEST_ID: _generate_request_id()}
