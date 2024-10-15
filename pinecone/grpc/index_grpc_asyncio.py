from typing import Optional, Union, List, Awaitable

from tqdm.asyncio import tqdm
from asyncio import Semaphore

from .vector_factory_grpc import VectorFactoryGRPC

from pinecone.core.grpc.protos.vector_service_pb2 import (
    Vector as GRPCVector,
    QueryVector as GRPCQueryVector,
    UpsertRequest,
    UpsertResponse,
    SparseValues as GRPCSparseValues,
)
from .base import GRPCIndexBase
from pinecone import Vector as NonGRPCVector
from pinecone.core.grpc.protos.vector_service_pb2_grpc import VectorServiceStub
from pinecone.utils import parse_non_empty_args

from .config import GRPCClientConfig
from pinecone.config import Config
from grpc._channel import Channel

__all__ = ["GRPCIndexAsyncio", "GRPCVector", "GRPCQueryVector", "GRPCSparseValues"]


class GRPCIndexAsyncio(GRPCIndexBase):
    """A client for interacting with a Pinecone index over GRPC with asyncio."""

    def __init__(
        self,
        index_name: str,
        config: Config,
        channel: Optional[Channel] = None,
        grpc_config: Optional[GRPCClientConfig] = None,
        _endpoint_override: Optional[str] = None,
    ):
        super().__init__(
            index_name=index_name,
            config=config,
            channel=channel,
            grpc_config=grpc_config,
            _endpoint_override=_endpoint_override,
            use_asyncio=True,
        )

    @property
    def stub_class(self):
        return VectorServiceStub

    async def upsert(
        self,
        vectors: Union[List[GRPCVector], List[NonGRPCVector], List[tuple], List[dict]],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Awaitable[UpsertResponse]:
        timeout = kwargs.pop("timeout", None)
        vectors = list(map(VectorFactoryGRPC.build, vectors))

        if batch_size is None:
            return await self._upsert_batch(vectors, namespace, timeout=timeout, **kwargs)

        else:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")

            semaphore = Semaphore(25)
            vector_batches = [
                vectors[i : i + batch_size] for i in range(0, len(vectors), batch_size)
            ]
            tasks = [
                self._upsert_batch(
                    vectors=batch, namespace=namespace, timeout=100, semaphore=semaphore
                )
                for batch in vector_batches
            ]

            return await tqdm.gather(*tasks, disable=not show_progress, desc="Upserted batches")

    async def _upsert_batch(
        self,
        vectors: List[GRPCVector],
        namespace: Optional[str],
        timeout: Optional[int] = None,
        semaphore: Optional[Semaphore] = None,
        **kwargs,
    ) -> Awaitable[UpsertResponse]:
        args_dict = parse_non_empty_args([("namespace", namespace)])
        request = UpsertRequest(vectors=vectors, **args_dict)
        if semaphore is not None:
            async with semaphore:
                return await self.runner.run_asyncio(
                    self.stub.Upsert, request, timeout=timeout, **kwargs
                )
        else:
            return await self.runner.run_asyncio(
                self.stub.Upsert, request, timeout=timeout, **kwargs
            )
