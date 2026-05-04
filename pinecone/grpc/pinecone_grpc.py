"""Backwards-compatibility shim for legacy ``PineconeGRPC``.

Provides a thin subclass of :class:`pinecone.Pinecone` that exposes
a legacy ``Index(name, host, **kwargs)`` factory returning a
:class:`GrpcIndex` (i.e. the same as
``pc.index(name, host, grpc=True)``). Preserved so pre-rewrite
callers using ``from pinecone.grpc import PineconeGRPC`` keep
working. New code should use::

    pc = Pinecone(api_key=...)
    idx = pc.index(name="my-index", grpc=True)

:meta private:
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pinecone._client import Pinecone
from pinecone.errors.exceptions import PineconeValueError

if TYPE_CHECKING:
    from pinecone.grpc import GrpcIndex


class PineconeGRPC(Pinecone):
    """Legacy gRPC client. Subclass of :class:`Pinecone`; data-plane
    calls via :meth:`Index` use gRPC instead of HTTP.

    :meta private:
    """

    def Index(self, name: str = "", host: str = "", **kwargs: Any) -> GrpcIndex:  # noqa: N802
        if not name and not host:
            raise PineconeValueError("Either name or host must be specified")
        kwargs.pop("pool_threads", None)
        if kwargs:
            raise TypeError(
                f"PineconeGRPC.Index() got unexpected keyword arguments: {sorted(kwargs)!r}"
            )
        from pinecone.grpc import GrpcIndex as _GrpcIndex

        return cast(_GrpcIndex, self.index(name=name, host=host, grpc=True))


__all__ = ["PineconeGRPC"]
