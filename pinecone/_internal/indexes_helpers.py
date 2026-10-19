"""Shared validation and body-building helpers for sync/async Indexes.

These are pure functions (no I/O) extracted from the duplicated private
methods on ``Indexes`` and ``AsyncIndexes``, plus polling helpers that
encapsulate the describe-until-ready loop.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, TypedDict

from pinecone._internal.validation import require_non_empty

if TYPE_CHECKING:
    from pinecone.models.indexes.index import IndexModel
from pinecone.errors.exceptions import PineconeValueError, ValidationError
from pinecone.models.enums import DeletionProtection, Metric, VectorType
from pinecone.models.indexes.specs import ByocSpec, IntegratedSpec

_VALID_DELETION_PROTECTION = frozenset({"enabled", "disabled"})


def _validate_deletion_protection(deletion_protection: DeletionProtection | str) -> None:
    resolved = resolve_enum_value(deletion_protection)
    if resolved not in _VALID_DELETION_PROTECTION:
        raise ValidationError(
            f"deletion_protection must be one of {sorted(_VALID_DELETION_PROTECTION)}, "
            f"got {resolved!r}"
        )


class IndexKwargs(TypedDict):
    """Typed kwargs for constructing :class:`~pinecone.index.Index` or
    :class:`~pinecone.async_client.async_index.AsyncIndex`.
    """

    host: str
    api_key: str
    additional_headers: dict[str, str]
    timeout: float
    proxy_url: str
    proxy_headers: dict[str, str]
    ssl_ca_certs: str | None
    ssl_verify: bool
    source_tag: str
    connection_pool_maxsize: int


class _LegacyIndexKwargs(IndexKwargs, total=False):
    """IndexKwargs extended with the legacy pool_threads field (sync client only)."""

    pool_threads: int


def resolve_enum_value(value: Any) -> Any:
    """Extract ``.value`` from enum-like objects, pass through otherwise."""
    return value.value if hasattr(value, "value") else value


def lowercase_metric(value: Any) -> Any:
    """Return a similarity metric in the lowercase form the server writes with.

    The server accepts a mixed-case metric such as ``"EUCLIDEAN"`` when an
    index is created but cannot derive a data schema from it, so every write
    to that index then fails. An enum member is resolved to its value first;
    anything that is not a string passes through for the server to judge.
    """
    resolved = resolve_enum_value(value)
    return resolved.lower() if isinstance(resolved, str) else resolved


def lowercase_schema_metrics(fields: Any) -> None:
    """Lowercase, in place, every ``metric`` a ``schema.fields`` mapping declares.

    Covers the dense vector field's own ``metric`` and the ``metric`` inside a
    string field's ``embed`` config. Anything that is not shaped like a field
    is left alone.
    """
    if not isinstance(fields, dict):
        return
    for field in fields.values():
        if not isinstance(field, dict):
            continue
        if "metric" in field:
            field["metric"] = lowercase_metric(field["metric"])
        embed = field.get("embed")
        if isinstance(embed, dict) and "metric" in embed:
            embed["metric"] = lowercase_metric(embed["metric"])


def validate_read_capacity(read_capacity: dict[str, Any]) -> None:
    """Validate read_capacity structure for index configure."""
    if "mode" not in read_capacity:
        raise ValidationError("read_capacity must contain a 'mode' key")

    if read_capacity["mode"] == "Dedicated":
        dedicated = read_capacity.get("dedicated")
        if dedicated is not None and not isinstance(dedicated, dict):
            raise ValidationError(
                "read_capacity with mode 'Dedicated' must contain a 'dedicated' dict"
            )
        if dedicated is not None and "scaling" in dedicated:
            manual = dedicated.get("manual")
            if manual is not None and not isinstance(manual, dict):
                raise ValidationError("dedicated read_capacity manual must be a dict")


def validate_byoc_inputs(
    *,
    name: str,
    spec: ByocSpec,
    dimension: int | None,
    deletion_protection: DeletionProtection | str,
) -> None:
    """Client-side validation for BYOC index creation."""
    require_non_empty("name", name)
    if len(name) > 45:
        raise ValidationError("index name must not exceed 45 characters")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValidationError("index name must contain only lowercase letters, digits, and hyphens")
    require_non_empty("environment", spec.environment)
    if dimension is None:
        raise ValidationError("dimension is required for BYOC indexes")

    _validate_deletion_protection(deletion_protection)


def build_byoc_body(
    *,
    name: str,
    spec: ByocSpec,
    dimension: int | None,
    metric: Metric | str,
    vector_type: VectorType | str,
    deletion_protection: DeletionProtection | str,
    tags: Mapping[str, str] | None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for POST /indexes (BYOC)."""
    body: dict[str, Any] = {
        "name": name,
        "metric": resolve_enum_value(metric),
        "vector_type": resolve_enum_value(vector_type),
        "deletion_protection": resolve_enum_value(deletion_protection),
    }
    if dimension is not None:
        body["dimension"] = dimension
    if tags is not None:
        body["tags"] = tags

    byoc_dict: dict[str, Any] = {"environment": spec.environment}
    if spec.read_capacity is not None:
        byoc_dict["read_capacity"] = spec.read_capacity
    if spec.schema is not None:
        byoc_dict["schema"] = _normalize_schema(spec.schema)
    if schema is not None:
        byoc_dict["schema"] = _normalize_schema(schema)
    body["spec"] = {"byoc": byoc_dict}

    return body


def validate_integrated_inputs(
    *,
    name: str,
    spec: IntegratedSpec,
    deletion_protection: DeletionProtection | str = "disabled",
) -> None:
    """Client-side validation for legacy integrated index creation."""
    require_non_empty("name", name)
    if len(name) > 45:
        raise ValidationError("index name must not exceed 45 characters")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValidationError("index name must contain only lowercase letters, digits, and hyphens")
    if not spec.cloud or not spec.cloud.strip():
        raise ValidationError("cloud is required for legacy integrated indexes")
    if not spec.embed.model or not spec.embed.model.strip():
        raise ValidationError("embed model is required for legacy integrated indexes")
    if not spec.embed.field_map:
        raise ValidationError("embed field_map is required for legacy integrated indexes")
    _validate_deletion_protection(deletion_protection)


def build_integrated_body(
    *,
    name: str,
    spec: IntegratedSpec,
    deletion_protection: DeletionProtection | str,
    tags: Mapping[str, str] | None,
    schema: dict[str, Any] | None = None,
    read_capacity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for POST /indexes (legacy integrated, model-backed)."""
    embed_body: dict[str, Any] = {
        "model": resolve_enum_value(spec.embed.model),
        "field_map": spec.embed.field_map,
    }
    if spec.embed.dimension is not None:
        embed_body["dimension"] = spec.embed.dimension
    if spec.embed.metric is not None:
        embed_body["metric"] = resolve_enum_value(spec.embed.metric)
    if spec.embed.read_parameters is not None:
        embed_body["read_parameters"] = spec.embed.read_parameters
    if spec.embed.write_parameters is not None:
        embed_body["write_parameters"] = spec.embed.write_parameters

    body: dict[str, Any] = {
        "name": name,
        "cloud": resolve_enum_value(spec.cloud),
        "region": spec.region,
        "embed": embed_body,
    }

    resolved_dp = resolve_enum_value(deletion_protection)
    if resolved_dp != "disabled":
        body["deletion_protection"] = resolved_dp
    if tags is not None:
        body["tags"] = tags
    if schema is not None:
        body["schema"] = _normalize_schema(schema)
    if read_capacity is not None:
        body["read_capacity"] = read_capacity

    return body


def _normalize_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Wrap a bare fields map in ``{"fields": ...}`` for backend compatibility.

    The backend's ``IndexMetadataMetadataSchema`` requires the top-level
    ``fields`` key. For ergonomic convenience, the SDK accepts either form:

    - ``{"fields": {"genre": {"filterable": True}}}`` (wrapped, per the OpenAPI spec)
    - ``{"genre": {"filterable": True}}`` (bare fields map)

    Both produce the wrapped wire format.
    """
    if list(raw.keys()) == ["fields"] and isinstance(raw.get("fields"), dict):
        return raw
    return {"fields": raw}


def validate_wait_timeout(timeout: int | None) -> None:
    """Reject a ``timeout`` that is neither ``None``, ``-1`` nor non-negative.

    Raises:
        PineconeValueError: If *timeout* is negative and not ``-1``.
    """
    if timeout is not None and timeout < 0 and timeout != -1:
        raise PineconeValueError(
            f"timeout must be None (wait indefinitely), -1 (do not wait), or a "
            f"non-negative number of seconds, got {timeout}. Pass -1 to skip waiting."
        )


def poll_sleep_seconds(poll_interval: float, timeout: float | None, start: float) -> float:
    """Return how long to sleep before the next poll without overshooting the deadline."""
    if timeout is None:
        return poll_interval
    remaining = timeout - (time.monotonic() - start)
    return max(0.0, min(poll_interval, remaining))


def poll_index_until_ready(
    describe_fn: Callable[[str], IndexModel],
    name: str,
    timeout: int | None,
    poll_interval: float = 5.0,
) -> IndexModel:
    """Poll ``describe_fn(name)`` until the index is ready or timeout is reached.

    Args:
        describe_fn: Synchronous callable that takes an index name and returns
            an :class:`IndexModel`.
        name: Name of the index to poll.
        timeout: Maximum seconds to wait. ``None`` means wait indefinitely.
        poll_interval: Seconds between successive polls.

    Returns:
        The :class:`IndexModel` once its status is ready.

    Raises:
        IndexInitFailedError: If the index enters ``InitializationFailed`` state.
        IndexTerminatedError: If the index enters ``Terminating`` or ``Disabled`` state.
        PineconeTimeoutError: If *timeout* seconds elapse without becoming ready.
    """
    from pinecone.errors.exceptions import (
        IndexInitFailedError,
        IndexTerminatedError,
        PineconeTimeoutError,
    )

    start = time.monotonic()
    while True:
        idx = describe_fn(name)
        if idx.status.ready:
            return idx
        if idx.status.state == "InitializationFailed":
            raise IndexInitFailedError(name)
        if idx.status.state in ("Terminating", "Disabled"):
            raise IndexTerminatedError(name, idx.status.state)
        if timeout is not None:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise PineconeTimeoutError(f"Index '{name}' not ready after {timeout}s")
        time.sleep(poll_sleep_seconds(poll_interval, timeout, start))


async def async_poll_index_until_ready(
    describe_fn: Callable[[str], Awaitable[IndexModel]],
    name: str,
    timeout: int | None,
    poll_interval: float = 5.0,
) -> IndexModel:
    """Async variant of :func:`poll_index_until_ready`.

    Args:
        describe_fn: Async callable that takes an index name and returns
            an :class:`IndexModel`.
        name: Name of the index to poll.
        timeout: Maximum seconds to wait. ``None`` means wait indefinitely.
        poll_interval: Seconds between successive polls.

    Returns:
        The :class:`IndexModel` once its status is ready.

    Raises:
        IndexInitFailedError: If the index enters ``InitializationFailed`` state.
        IndexTerminatedError: If the index enters ``Terminating`` or ``Disabled`` state.
        PineconeTimeoutError: If *timeout* seconds elapse without becoming ready.
    """
    from pinecone.errors.exceptions import (
        IndexInitFailedError,
        IndexTerminatedError,
        PineconeTimeoutError,
    )

    start = time.monotonic()
    while True:
        idx = await describe_fn(name)
        if idx.status.ready:
            return idx
        if idx.status.state == "InitializationFailed":
            raise IndexInitFailedError(name)
        if idx.status.state in ("Terminating", "Disabled"):
            raise IndexTerminatedError(name, idx.status.state)
        if timeout is not None:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise PineconeTimeoutError(f"Index '{name}' not ready after {timeout}s")
        await asyncio.sleep(poll_sleep_seconds(poll_interval, timeout, start))
