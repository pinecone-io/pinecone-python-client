from __future__ import annotations

import logging
import warnings
from typing import Dict, TYPE_CHECKING, Any
from typing_extensions import Self

from pinecone.config import PineconeConfig, ConfigBuilder

from pinecone.utils import normalize_host, require_kwargs, docslinks

from .pinecone_interface_asyncio import PineconeAsyncioDBControlInterface
from .pinecone import check_realistic_host

if TYPE_CHECKING:
    from pinecone.db_control.types import ConfigureIndexEmbed, CreateIndexForModelEmbedTypedDict
    from pinecone.db_data import _IndexAsyncio
    from pinecone.inference import AsyncioInference
    from pinecone.db_control.db_control_asyncio import DBControlAsyncio
    from pinecone.db_control.enums import (
        Metric,
        VectorType,
        DeletionProtection,
        PodType,
        CloudProvider,
        AwsRegion,
        GcpRegion,
        AzureRegion,
    )
    from pinecone.db_control.models import (
        ServerlessSpec,
        PodSpec,
        ByocSpec,
        IndexModel,
        IndexList,
        CollectionList,
        IndexEmbed,
        BackupModel,
        BackupList,
        RestoreJobModel,
        RestoreJobList,
    )
    from pinecone.db_control.models.serverless_spec import (
        ReadCapacityDict,
        MetadataSchemaFieldConfig,
    )
    from pinecone.core.openapi.db_control.model.read_capacity import ReadCapacity
    from pinecone.core.openapi.db_control.model.read_capacity_on_demand_spec import (
        ReadCapacityOnDemandSpec,
    )
    from pinecone.core.openapi.db_control.model.read_capacity_dedicated_spec import (
        ReadCapacityDedicatedSpec,
    )
    from pinecone.core.openapi.db_control.model.backup_model_schema import BackupModelSchema
    from pinecone.core.openapi.db_control.api.manage_indexes_api import AsyncioManageIndexesApi
    from pinecone.db_control.index_host_store import IndexHostStore

logger = logging.getLogger(__name__)
""" :meta private: """


class PineconeAsyncio(PineconeAsyncioDBControlInterface):
    """
    ``PineconeAsyncio`` is an asyncio client for interacting with Pinecone's control plane API.

    This class implements methods for managing and interacting with Pinecone resources
    such as collections and indexes.

    To perform data operations such as inserting and querying vectors, use the ``IndexAsyncio`` class.

    .. code-block:: python

        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio() as pc:
                async with pc.IndexAsyncio(host="my-index.pinecone.io") as idx:
                    await idx.upsert(vectors=[(1, [1, 2, 3]), (2, [4, 5, 6])])

        asyncio.run(main())

    """

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        proxy_url: str | None = None,
        # proxy_headers: dict[str, str] | None = None,
        ssl_ca_certs: str | None = None,
        ssl_verify: bool | None = None,
        additional_headers: dict[str, str] | None = {},
        **kwargs,
    ) -> None:
        """
        Initialize the ``PineconeAsyncio`` client.

        :param api_key: The API key to use for authentication. If not passed via kwarg, the API key will be read from the environment variable ``PINECONE_API_KEY``.
        :type api_key: str, optional
        :param host: The control plane host. If unspecified, the host ``api.pinecone.io`` will be used.
        :type host: str, optional
        :param proxy_url: The URL of the proxy to use for the connection.
        :type proxy_url: str, optional
        :param ssl_ca_certs: The path to the SSL CA certificate bundle to use for the connection. This path should point to a file in PEM format. When not passed, the SDK will use the certificate bundle returned from ``certifi.where()``.
        :type ssl_ca_certs: str, optional
        :param ssl_verify: SSL verification is performed by default, but can be disabled using the boolean flag when testing with Pinecone Local or troubleshooting a proxy setup. You should never run with SSL verification disabled in production.
        :type ssl_verify: bool, optional
        :param additional_headers: Additional headers to pass to the API. This is mainly to support internal testing at Pinecone. End users should not need to use this unless following specific instructions to do so.
        :type additional_headers: dict[str, str], optional

        .. note::

            The ``proxy_headers`` parameter is not currently supported for ``PineconeAsyncio``.

        """
        for deprecated_kwarg in {"config", "openapi_config"}:
            if deprecated_kwarg in kwargs:
                raise NotImplementedError(
                    f"Passing {deprecated_kwarg} is no longer supported. Please pass individual settings such as proxy_url, ssl_ca_certs, and ssl_verify directly to the Pinecone constructor as keyword arguments. See the README at {docslinks['README']} for examples."
                )

        for unimplemented_kwarg in {"proxy_headers"}:
            if unimplemented_kwarg in kwargs:
                raise NotImplementedError(
                    f"You have passed {unimplemented_kwarg} but this configuration has not been implemented for PineconeAsyncio."
                )

        self._config = PineconeConfig.build(
            api_key=api_key,
            host=host,
            additional_headers=additional_headers,
            proxy_url=proxy_url,
            proxy_headers=None,
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
            **kwargs,
        )
        """ :meta private: """

        self._openapi_config = ConfigBuilder.build_openapi_config(self._config, **kwargs)
        """ :meta private: """

        self._inference: "AsyncioInference" | None = None  # Lazy initialization
        """ :meta private: """

        self._db_control: "DBControlAsyncio" | None = None  # Lazy initialization
        """ :meta private: """

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_value: BaseException | None, traceback: Any | None
    ) -> bool | None:
        await self.close()
        return None

    async def close(self) -> None:
        """Cleanup resources used by the Pinecone client.

        This method should be called when the client is no longer needed so that
        it can cleanup the aiohttp session and other resources.

        After close has been called, the client instance should not be used.

        .. code-block:: python

            import asyncio
            from pinecone import PineconeAsyncio

            async def main():
                pc = PineconeAsyncio()
                desc = await pc.describe_index(name="my-index")
                await pc.close()

            asyncio.run(main())

        If you are using the client as a context manager, the close method is called automatically
        when exiting.

        .. code-block:: python

            import asyncio
            from pinecone import PineconeAsyncio

            async def main():
                async with PineconeAsyncio() as pc:
                    desc = await pc.describe_index(name="my-index")

                # No need to call close in this case because the "async with" syntax
                # automatically calls close when exiting the block.

            asyncio.run(main())

        """
        await self.db._index_api.api_client.close()

    @property
    def inference(self) -> "AsyncioInference":
        """Dynamically create and cache the AsyncioInference instance."""
        if self._inference is None:
            from pinecone.inference import AsyncioInference

            self._inference = AsyncioInference(api_client=self.db._index_api.api_client)
        return self._inference

    @property
    def db(self) -> "DBControlAsyncio":
        """
        db is a namespace where an instance of the ``pinecone.db_control.DBControlAsyncio`` class is lazily created and cached.
        """
        if self._db_control is None:
            from .db_control.db_control_asyncio import DBControlAsyncio

            self._db_control = DBControlAsyncio(
                config=self._config, openapi_config=self._openapi_config
            )
        return self._db_control

    @property
    def index_host_store(self) -> "IndexHostStore":
        """:meta private:"""
        warnings.warn(
            "The `index_host_store` property is deprecated. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        # IndexResourceAsyncio doesn't have _index_host_store, access the singleton directly
        from pinecone.db_control.index_host_store import IndexHostStore

        return IndexHostStore()

    @property
    def index_api(self) -> "AsyncioManageIndexesApi":
        """:meta private:"""
        warnings.warn(
            "The `index_api` property is deprecated. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.db._index_api

    async def create_index(
        self,
        name: str,
        spec: Dict | "ServerlessSpec" | "PodSpec" | "ByocSpec",
        dimension: int | None = None,
        metric: ("Metric" | str) | None = "cosine",
        timeout: int | None = None,
        deletion_protection: ("DeletionProtection" | str) | None = "disabled",
        vector_type: ("VectorType" | str) | None = "dense",
        tags: dict[str, str] | None = None,
    ) -> "IndexModel":
        resp = await self.db.index.create(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric,
            deletion_protection=deletion_protection,
            vector_type=vector_type,
            tags=tags,
            timeout=timeout,
        )
        return resp

    async def create_index_for_model(
        self,
        name: str,
        cloud: "CloudProvider" | str,
        region: "AwsRegion" | "GcpRegion" | "AzureRegion" | str,
        embed: "IndexEmbed" | "CreateIndexForModelEmbedTypedDict",
        tags: dict[str, str] | None = None,
        deletion_protection: ("DeletionProtection" | str) | None = "disabled",
        read_capacity: (
            "ReadCapacityDict"
            | "ReadCapacity"
            | "ReadCapacityOnDemandSpec"
            | "ReadCapacityDedicatedSpec"
        )
        | None = None,
        schema: (
            dict[
                str, "MetadataSchemaFieldConfig"
            ]  # Direct field mapping: {field_name: {filterable: bool}}
            | dict[
                str, dict[str, Any]
            ]  # Dict with "fields" wrapper: {"fields": {field_name: {...}}, ...}
            | "BackupModelSchema"  # OpenAPI model instance
        )
        | None = None,
        timeout: int | None = None,
    ) -> "IndexModel":
        return await self.db.index.create_for_model(
            name=name,
            cloud=cloud,
            region=region,
            embed=embed,
            tags=tags,
            deletion_protection=deletion_protection,
            read_capacity=read_capacity,
            schema=schema,
            timeout=timeout,
        )

    @require_kwargs
    async def create_index_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: ("DeletionProtection" | str) | None = "disabled",
        tags: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> "IndexModel":
        return await self.db.index.create_from_backup(
            name=name,
            backup_id=backup_id,
            deletion_protection=deletion_protection,
            tags=tags,
            timeout=timeout,
        )

    async def delete_index(self, name: str, timeout: int | None = None) -> None:
        return await self.db.index.delete(name=name, timeout=timeout)

    async def list_indexes(self) -> "IndexList":
        return await self.db.index.list()

    async def describe_index(self, name: str) -> "IndexModel":
        return await self.db.index.describe(name=name)

    async def has_index(self, name: str) -> bool:
        return await self.db.index.has(name=name)

    async def configure_index(
        self,
        name: str,
        replicas: int | None = None,
        pod_type: ("PodType" | str) | None = None,
        deletion_protection: ("DeletionProtection" | str) | None = None,
        tags: dict[str, str] | None = None,
        embed: ("ConfigureIndexEmbed" | Dict) | None = None,
        read_capacity: (
            "ReadCapacityDict"
            | "ReadCapacity"
            | "ReadCapacityOnDemandSpec"
            | "ReadCapacityDedicatedSpec"
        )
        | None = None,
    ) -> None:
        return await self.db.index.configure(
            name=name,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
            embed=embed,
            read_capacity=read_capacity,
        )

    async def create_collection(self, name: str, source: str) -> None:
        return await self.db.collection.create(name=name, source=source)

    async def list_collections(self) -> "CollectionList":
        return await self.db.collection.list()

    async def delete_collection(self, name: str) -> None:
        return await self.db.collection.delete(name=name)

    async def describe_collection(self, name: str) -> dict[str, Any]:
        return await self.db.collection.describe(name=name)

    @require_kwargs
    async def create_backup(
        self, *, index_name: str, backup_name: str, description: str = ""
    ) -> "BackupModel":
        return await self.db.backup.create(
            index_name=index_name, backup_name=backup_name, description=description
        )

    @require_kwargs
    async def list_backups(
        self,
        *,
        index_name: str | None = None,
        limit: int | None = 10,
        pagination_token: str | None = None,
    ) -> "BackupList":
        return await self.db.backup.list(
            index_name=index_name, limit=limit, pagination_token=pagination_token
        )

    @require_kwargs
    async def describe_backup(self, *, backup_id: str) -> "BackupModel":
        return await self.db.backup.describe(backup_id=backup_id)

    @require_kwargs
    async def delete_backup(self, *, backup_id: str) -> None:
        return await self.db.backup.delete(backup_id=backup_id)

    @require_kwargs
    async def list_restore_jobs(
        self, *, limit: int | None = 10, pagination_token: str | None = None
    ) -> "RestoreJobList":
        return await self.db.restore_job.list(limit=limit, pagination_token=pagination_token)

    @require_kwargs
    async def describe_restore_job(self, *, job_id: str) -> "RestoreJobModel":
        return await self.db.restore_job.describe(job_id=job_id)

    def IndexAsyncio(self, host: str, **kwargs) -> "_IndexAsyncio":
        from pinecone.db_data import _IndexAsyncio

        api_key = self._config.api_key
        openapi_config = self._openapi_config

        if host is None or host == "":
            raise ValueError("A host must be specified")

        check_realistic_host(host)
        index_host = normalize_host(host)

        return _IndexAsyncio(
            host=index_host,
            api_key=api_key,
            openapi_config=openapi_config,
            source_tag=self._config.source_tag,
            **kwargs,
        )
