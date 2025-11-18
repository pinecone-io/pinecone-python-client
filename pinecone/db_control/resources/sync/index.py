from __future__ import annotations

import time
import logging
from typing import Dict, TYPE_CHECKING, Any

from pinecone.db_control.index_host_store import IndexHostStore

from pinecone.db_control.models import IndexModel, IndexList, IndexEmbed
from pinecone.utils import docslinks, require_kwargs, PluginAware

from pinecone.db_control.types import CreateIndexForModelEmbedTypedDict
from pinecone.db_control.request_factory import PineconeDBControlRequestFactory
from pinecone.core.openapi.db_control import API_VERSION
from pinecone.db_control.types.configure_index_embed import ConfigureIndexEmbed

logger = logging.getLogger(__name__)
""" :meta private: """

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
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
    from pinecone.db_control.models import ServerlessSpec, PodSpec, ByocSpec, IndexEmbed
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


class IndexResource(PluginAware):
    def __init__(
        self,
        index_api: "ManageIndexesApi",
        config: "Config",
        openapi_config: "OpenApiConfiguration",
        pool_threads: int,
    ):
        self._index_api = index_api
        """ :meta private: """

        self.config = config
        """ :meta private: """

        self._openapi_config = openapi_config
        """ :meta private: """

        self._pool_threads = pool_threads
        """ :meta private: """

        self._index_host_store = IndexHostStore()
        """ :meta private: """

        super().__init__()  # Initialize PluginAware

    @require_kwargs
    def create(
        self,
        *,
        name: str,
        spec: Dict | "ServerlessSpec" | "PodSpec" | "ByocSpec",
        dimension: int | None = None,
        metric: ("Metric" | str) | None = "cosine",
        timeout: int | None = None,
        deletion_protection: ("DeletionProtection" | str) | None = "disabled",
        vector_type: ("VectorType" | str) | None = "dense",
        tags: dict[str, str] | None = None,
    ) -> IndexModel:
        req = PineconeDBControlRequestFactory.create_index_request(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric,
            deletion_protection=deletion_protection,
            vector_type=vector_type,
            tags=tags,
        )
        resp = self._index_api.create_index(create_index_request=req)

        if timeout == -1:
            from typing import cast

            return IndexModel(cast(Any, resp))
        return self.__poll_describe_index_until_ready(name, timeout)

    @require_kwargs
    def create_for_model(
        self,
        *,
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
    ) -> IndexModel:
        req = PineconeDBControlRequestFactory.create_index_for_model_request(
            name=name,
            cloud=cloud,
            region=region,
            embed=embed,
            tags=tags,
            deletion_protection=deletion_protection,
            read_capacity=read_capacity,
            schema=schema,
        )
        resp = self._index_api.create_index_for_model(req)

        if timeout == -1:
            from typing import cast

            return IndexModel(cast(Any, resp))
        return self.__poll_describe_index_until_ready(name, timeout)

    @require_kwargs
    def create_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: ("DeletionProtection" | str) | None = "disabled",
        tags: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> IndexModel:
        """
        Create an index from a backup.

        Args:
            name (str): The name of the index to create.
            backup_id (str): The ID of the backup to create the index from.
            deletion_protection (DeletionProtection): The deletion protection to use for the index.
            tags (dict[str, str]): The tags to use for the index.
            timeout (int): The number of seconds to wait for the index to be ready. If -1, the function will return without polling for the index status to be ready. If None, the function will poll indefinitely for the index to be ready.

        Returns:
            IndexModel: The created index.
        """
        req = PineconeDBControlRequestFactory.create_index_from_backup_request(
            name=name, deletion_protection=deletion_protection, tags=tags
        )
        resp = self._index_api.create_index_from_backup_operation(
            backup_id=backup_id, create_index_from_backup_request=req
        )
        logger.info(f"Creating index from backup. Response: {resp}")

        if timeout == -1:
            return self.describe(name=name)
        return self.__poll_describe_index_until_ready(name, timeout)

    def __poll_describe_index_until_ready(
        self, name: str, timeout: int | None = None
    ) -> IndexModel:
        total_wait_time = 0
        while True:
            description = self.describe(name=name)
            if description.status.state == "InitializationFailed":
                raise Exception(
                    f"Index {name} failed to initialize. The index status is {description.status.state}."
                )
            if description.status.ready:
                return description

            if timeout is not None and total_wait_time >= timeout:
                logger.error(
                    f"Index {name} is not ready after {total_wait_time} seconds. Timeout reached."
                )
                link = docslinks["API_DESCRIBE_INDEX"](API_VERSION)
                timeout_msg = f"Index {name} is not ready after {total_wait_time} seconds. Please call describe_index() to confirm index status. See docs at {link}"
                raise TimeoutError(timeout_msg)

            logger.debug(
                f"Waiting for index {name} to be ready. Total wait time {total_wait_time} seconds."
            )

            total_wait_time += 5
            time.sleep(5)

    @require_kwargs
    def delete(self, *, name: str, timeout: int | None = None) -> None:
        self._index_api.delete_index(name)
        self._index_host_store.delete_host(self.config, name)

        if timeout == -1:
            return

        if timeout is None:
            while self.has(name=name):
                time.sleep(5)
        else:
            while self.has(name=name) and timeout >= 0:
                time.sleep(5)
                timeout -= 5
        if timeout and timeout < 0:
            raise (
                TimeoutError(
                    "Please call the list_indexes API ({}) to confirm if index is deleted".format(
                        "https://www.pinecone.io/docs/api/operation/list_indexes/"
                    )
                )
            )

    @require_kwargs
    def list(self) -> IndexList:
        response = self._index_api.list_indexes()
        return IndexList(response)

    @require_kwargs
    def describe(self, *, name: str) -> IndexModel:
        api_instance = self._index_api
        description = api_instance.describe_index(name)
        host = description.host
        self._index_host_store.set_host(self.config, name, host)

        return IndexModel(description)

    @require_kwargs
    def has(self, *, name: str) -> bool:
        if name in self.list().names():
            return True
        else:
            return False

    @require_kwargs
    def configure(
        self,
        *,
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
        api_instance = self._index_api
        description = self.describe(name=name)

        req = PineconeDBControlRequestFactory.configure_index_request(
            description=description,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
            embed=embed,
            read_capacity=read_capacity,
        )
        api_instance.configure_index(name, configure_index_request=req)

    def _get_host(self, name: str) -> str:
        """:meta private:"""
        return self._index_host_store.get_host(
            api=self._index_api, config=self.config, index_name=name
        )
