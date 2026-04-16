"""Async preview indexes namespace — control-plane operations backed by 2026-01.alpha."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import msgspec
import orjson

from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.errors.exceptions import PineconeValueError
from pinecone.preview._internal.adapters.indexes import create_adapter
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.models.indexes import PreviewIndexModel
from pinecone.preview.models.requests import PreviewCreateIndexRequest
from pinecone.preview.models.schema import PreviewSchema

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig

logger = logging.getLogger(__name__)

__all__ = ["AsyncPreviewIndexes"]


class AsyncPreviewIndexes:
    """Async control-plane operations for preview indexes (``2026-01.alpha``).

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Access via ``pc.preview.indexes``.

    Args:
        config: SDK configuration used to construct an async HTTP client targeting
            the preview API version.

    Examples::

        async def main():
            pc = AsyncPinecone(api_key="...")
            index = await pc.preview.indexes.create(
                schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536}}},
                name="my-preview-index",
            )
    """

    def __init__(self, config: PineconeConfig) -> None:
        from pinecone._internal.config import PineconeConfig as _PineconeConfig
        from pinecone._internal.http_client import AsyncHTTPClient as _AsyncHTTPClient

        self._config = config
        cp_host = (config.host or DEFAULT_BASE_URL).rstrip("/")
        cp_config = _PineconeConfig(
            api_key=config.api_key,
            host=cp_host,
            timeout=config.timeout,
            additional_headers=config.additional_headers,
            source_tag=config.source_tag or "",
            proxy_url=config.proxy_url or "",
            proxy_headers=config.proxy_headers,
            ssl_ca_certs=config.ssl_ca_certs,
            ssl_verify=config.ssl_verify,
            connection_pool_maxsize=config.connection_pool_maxsize,
            retry_config=config.retry_config,
        )
        self._http = _AsyncHTTPClient(cp_config, INDEXES_API_VERSION)

    def __repr__(self) -> str:
        return "AsyncPreviewIndexes()"

    async def create(
        self,
        *,
        schema: dict[str, Any],
        name: str | None = None,
        deployment: dict[str, Any] | None = None,
        read_capacity: dict[str, Any] | None = None,
        deletion_protection: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> PreviewIndexModel:
        """Create a new preview index.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            schema: Index schema definition. A dict with a ``"fields"`` key
                mapping field names to their typed field definitions. Example::

                    {
                        "fields": {
                            "embedding": {"type": "dense_vector", "dimension": 1536},
                            "title": {"type": "string", "full_text_searchable": True},
                        }
                    }

            name: Optional name for the index. If omitted, the API assigns one.
            deployment: Optional deployment configuration dict. Must include a
                ``"deployment_type"`` key (e.g. ``"managed"``, ``"pod"``,
                ``"byoc"``). When omitted the API uses its default deployment.
            read_capacity: Optional read capacity configuration dict. Must
                include a ``"mode"`` key (``"OnDemand"`` or ``"Dedicated"``).
                When omitted the API uses on-demand mode.
            deletion_protection: Optional deletion-protection setting —
                ``"enabled"`` or ``"disabled"``. Defaults to ``"disabled"``
                if not provided.
            tags: Optional key-value tags for the index. Keys must be at most
                80 characters; values must be at most 120 characters.

        Returns:
            :class:`PreviewIndexModel` describing the newly created index. The
            returned model's ``status.ready`` may be ``False`` — use
            ``describe()`` to poll until the index is ready.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If a tag key
                exceeds 80 characters or a tag value exceeds 120 characters.
            :exc:`msgspec.ValidationError`: If ``schema`` cannot be converted
                to the expected typed model.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the API returns an
                error response.

        Examples::

            async def main():
                index = await pc.preview.indexes.create(
                    schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 1536}}},
                    name="my-preview-index",
                )
        """
        if tags is not None:
            for key, value in tags.items():
                if len(key) > 80:
                    raise PineconeValueError(f"Tag key {key!r} exceeds the 80-character limit.")
                if len(value) > 120:
                    raise PineconeValueError(
                        f"Tag value for key {key!r} exceeds the 120-character limit."
                    )

        typed_schema = msgspec.convert(schema, PreviewSchema)

        req = PreviewCreateIndexRequest(
            schema=typed_schema,
            name=name,
            deployment=deployment,
            read_capacity=read_capacity,
            deletion_protection=deletion_protection,
            tags=tags,
        )

        logger.info("Creating preview index name=%r", name)
        response = await self._http.post(
            "/indexes",
            content=create_adapter.to_request(req),
            headers={"Content-Type": "application/json"},
        )
        return create_adapter.from_response(orjson.loads(response.content))
