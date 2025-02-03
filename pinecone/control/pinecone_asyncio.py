import logging
import asyncio
from typing import Optional, Dict, Union

from pinecone.config import PineconeConfig, ConfigBuilder

from pinecone.core.openapi.db_control.api.manage_indexes_api import AsyncioManageIndexesApi
from pinecone.openapi_support import AsyncioApiClient

from pinecone.utils import normalize_host, setup_async_openapi_client
from pinecone.core.openapi.db_control import API_VERSION
from pinecone.models import (
    ServerlessSpec,
    PodSpec,
    IndexModel,
    IndexList,
    CollectionList,
    IndexEmbed,
)
from pinecone.utils import docslinks

from pinecone.data import _AsyncioIndex, _AsyncioInference
from pinecone.enums import (
    Metric,
    VectorType,
    DeletionProtection,
    PodType,
    CloudProvider,
    AwsRegion,
    GcpRegion,
    AzureRegion,
)
from .types import CreateIndexForModelEmbedTypedDict
from .request_factory import PineconeDBControlRequestFactory
from .pinecone_interface_asyncio import PineconeAsyncioDBControlInterface
from .pinecone import check_realistic_host

logger = logging.getLogger(__name__)
""" @private """


class PineconeAsyncio(PineconeAsyncioDBControlInterface):
    """
    An asyncio client for interacting with Pinecone's vector database.

    This class implements methods for managing and interacting with Pinecone resources
    such as collections and indexes.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        # proxy_url: Optional[str] = None,
        # proxy_headers: Optional[Dict[str, str]] = None,
        # ssl_ca_certs: Optional[str] = None,
        # ssl_verify: Optional[bool] = None,
        # additional_headers: Optional[Dict[str, str]] = {},
        **kwargs,
    ):
        for kwarg in {
            "additional_headers",
            "proxy_url",
            "proxy_headers",
            "ssl_ca_certs",
            "ssl_verify",
        }:
            if kwarg in kwargs:
                raise NotImplementedError(
                    f"You have passed {kwarg} but this configuration has not been implemented yet for PineconeAsyncio."
                )

        self.config = PineconeConfig.build(
            api_key=api_key,
            host=host,
            additional_headers=None,
            proxy_url=None,
            proxy_headers=None,
            ssl_ca_certs=None,
            ssl_verify=None,
            **kwargs,
        )
        self.openapi_config = ConfigBuilder.build_openapi_config(self.config, **kwargs)

        self._inference = None  # Lazy initialization

        self.index_api = setup_async_openapi_client(
            api_client_klass=AsyncioApiClient,
            api_klass=AsyncioManageIndexesApi,
            config=self.config,
            openapi_config=self.openapi_config,
            api_version=API_VERSION,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def close(self):
        await self.index_api.api_client.close()

    @property
    def inference(self):
        """Dynamically create and cache the Inference instance."""
        if self._inference is None:
            self._inference = _AsyncioInference(api_client=self.index_api.api_client)
        return self._inference

    async def create_index(
        self,
        name: str,
        spec: Union[Dict, ServerlessSpec, PodSpec],
        dimension: Optional[int] = None,
        metric: Optional[Union[Metric, str]] = Metric.COSINE,
        timeout: Optional[int] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        vector_type: Optional[Union[VectorType, str]] = VectorType.DENSE,
        tags: Optional[Dict[str, str]] = None,
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
        resp = await self.index_api.create_index(create_index_request=req)

        if timeout == -1:
            return IndexModel(resp)
        return await self.__poll_describe_index_until_ready(name, timeout)

    async def create_index_for_model(
        self,
        name: str,
        cloud: Union[CloudProvider, str],
        region: Union[AwsRegion, GcpRegion, AzureRegion, str],
        embed: Union[IndexEmbed, CreateIndexForModelEmbedTypedDict],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        timeout: Optional[int] = None,
    ) -> IndexModel:
        req = PineconeDBControlRequestFactory.create_index_for_model_request(
            name=name,
            cloud=cloud,
            region=region,
            embed=embed,
            tags=tags,
            deletion_protection=deletion_protection,
        )
        resp = await self.index_api.create_index_for_model(req)

        if timeout == -1:
            return IndexModel(resp)
        return await self.__poll_describe_index_until_ready(name, timeout)

    async def __poll_describe_index_until_ready(self, name: str, timeout: Optional[int] = None):
        description = None

        async def is_ready() -> bool:
            nonlocal description
            description = await self.describe_index(name=name)
            return description.status.ready

        total_wait_time = 0
        if timeout is None:
            # Wait indefinitely
            while not await is_ready():
                logger.debug(
                    f"Waiting for index {name} to be ready. Total wait time {total_wait_time} seconds."
                )
                total_wait_time += 5
                await asyncio.sleep(5)

        else:
            # Wait for a maximum of timeout seconds
            while not await is_ready():
                if timeout < 0:
                    logger.error(f"Index {name} is not ready. Timeout reached.")
                    link = docslinks["API_DESCRIBE_INDEX"]
                    timeout_msg = (
                        f"Please call describe_index() to confirm index status. See docs at {link}"
                    )
                    raise TimeoutError(timeout_msg)

                logger.debug(
                    f"Waiting for index {name} to be ready. Total wait time: {total_wait_time}"
                )
                total_wait_time += 5
                await asyncio.sleep(5)
                timeout -= 5

        return description

    async def delete_index(self, name: str, timeout: Optional[int] = None):
        await self.index_api.delete_index(name)

        if timeout == -1:
            return

        if timeout is None:
            while await self.has_index(name):
                await asyncio.sleep(5)
        else:
            while await self.has_index(name) and timeout >= 0:
                await asyncio.sleep(5)
                timeout -= 5
        if timeout and timeout < 0:
            raise (
                TimeoutError(
                    "Please call the list_indexes API ({}) to confirm if index is deleted".format(
                        "https://www.pinecone.io/docs/api/operation/list_indexes/"
                    )
                )
            )

    async def list_indexes(self) -> IndexList:
        response = await self.index_api.list_indexes()
        return IndexList(response)

    async def describe_index(self, name: str) -> IndexModel:
        description = await self.index_api.describe_index(name)
        return IndexModel(description)

    async def has_index(self, name: str) -> bool:
        available_indexes = await self.list_indexes()
        if name in available_indexes.names():
            return True
        else:
            return False

    async def configure_index(
        self,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[Union[PodType, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        description = await self.describe_index(name=name)

        req = PineconeDBControlRequestFactory.configure_index_request(
            description=description,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
        )
        await self.index_api.configure_index(name, configure_index_request=req)

    async def create_collection(self, name: str, source: str):
        req = PineconeDBControlRequestFactory.create_collection_request(name=name, source=source)
        await self.index_api.create_collection(create_collection_request=req)

    async def list_collections(self) -> CollectionList:
        response = await self.index_api.list_collections()
        return CollectionList(response)

    async def delete_collection(self, name: str):
        await self.index_api.delete_collection(name)

    async def describe_collection(self, name: str):
        return await self.index_api.describe_collection(name).to_dict()

    def Index(self, host: str, **kwargs):
        api_key = self.config.api_key
        openapi_config = self.openapi_config

        if host is None or host == "":
            raise ValueError("A host must be specified")

        check_realistic_host(host)
        index_host = normalize_host(host)

        return _AsyncioIndex(
            host=index_host,
            api_key=api_key,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs,
        )
