"""Protocol stub for the Rust-backed GrpcChannel extension module.

Declares the expected interface so mypy can verify all call sites in
:class:`~pinecone.grpc.GrpcIndex` even when the native extension is not
built.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GrpcChannelProtocol(Protocol):
    """Structural type for the Rust-backed ``GrpcChannel``."""

    def upsert(
        self,
        vectors: list[dict[str, Any]],
        namespace: str | None,
    ) -> dict[str, Any]:
        """Upsert a batch of vectors."""
        ...

    def query(
        self,
        top_k: int,
        *,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
        sparse_vector: dict[str, Any] | None = None,
        scan_factor: float | None = None,
        max_candidates: int | None = None,
    ) -> dict[str, Any]:
        """Query for nearest neighbors."""
        ...

    def fetch(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Fetch vectors by ID."""
        ...

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        delete_all: bool = False,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> None:
        """Delete vectors."""
        ...

    def update(
        self,
        id: str | None,
        *,
        values: list[float] | None = None,
        sparse_values: dict[str, Any] | None = None,
        set_metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
        dry_run: bool | None = None,
    ) -> dict[str, Any]:
        """Update a vector."""
        ...

    def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """List vector IDs."""
        ...

    def describe_index_stats(
        self,
        *,
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Describe index statistics."""
        ...

    def close(self) -> None:
        """Close the channel and release resources."""
        ...
