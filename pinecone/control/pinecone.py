import time
import logging
from typing import Optional, Dict, Union
from multiprocessing import cpu_count

from .index_host_store import IndexHostStore
from .pinecone_interface import PineconeDBControlInterface

from pinecone.config import PineconeConfig, ConfigBuilder

from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
from pinecone.openapi_support.api_client import ApiClient


from pinecone.utils import normalize_host, setup_openapi_client, PluginAware
from pinecone.core.openapi.db_control import API_VERSION
from pinecone.models import (
    ServerlessSpec,
    PodSpec,
    IndexModel,
    IndexList,
    CollectionList,
    IndexEmbed,
)
from .langchain_import_warnings import _build_langchain_attribute_error_message
from pinecone.utils import docslinks
from pinecone.data import _Index, _Inference, _IndexAsyncio

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

logger = logging.getLogger(__name__)
""" @private """


class Pinecone(PineconeDBControlInterface, PluginAware):
    """
    A client for interacting with Pinecone's vector database.

    This class implements methods for managing and interacting with Pinecone resources
    such as collections and indexes.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        proxy_url: Optional[str] = None,
        proxy_headers: Optional[Dict[str, str]] = None,
        ssl_ca_certs: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        pool_threads: Optional[int] = None,
        **kwargs,
    ):
        for deprecated_kwarg in {"config", "openapi_config", "index_api"}:
            if deprecated_kwarg in kwargs:
                raise NotImplementedError(
                    f"Passing {deprecated_kwarg} is no longer supported. Please pass individual settings such as proxy_url, proxy_headers, ssl_ca_certs, and ssl_verify directly to the Pinecone constructor as keyword arguments. See the README at {docslinks['README']} for examples."
                )

        self.config = PineconeConfig.build(
            api_key=api_key,
            host=host,
            additional_headers=additional_headers,
            proxy_url=proxy_url,
            proxy_headers=proxy_headers,
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
            **kwargs,
        )
        """ @private """

        self.openapi_config = ConfigBuilder.build_openapi_config(self.config, **kwargs)
        """ @private """

        if pool_threads is None:
            self.pool_threads = 5 * cpu_count()
            """ @private """
        else:
            self.pool_threads = pool_threads
            """ @private """

        self._inference = None  # Lazy initialization
        """ @private """

        self.index_api = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=ManageIndexesApi,
            config=self.config,
            openapi_config=self.openapi_config,
            pool_threads=pool_threads,
            api_version=API_VERSION,
        )
        """ @private """

        self.index_host_store = IndexHostStore()
        """ @private """

        self.load_plugins(
            config=self.config, openapi_config=self.openapi_config, pool_threads=self.pool_threads
        )

    @property
    def inference(self):
        """
        Inference is a namespace where an instance of the `pinecone.data.features.inference.inference.Inference` class is lazily created and cached.
        """
        if self._inference is None:
            self._inference = _Inference(config=self.config, openapi_config=self.openapi_config)
        return self._inference

    def create_index(
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
        resp = self.index_api.create_index(create_index_request=req)

        if timeout == -1:
            return IndexModel(resp)
        return self.__poll_describe_index_until_ready(name, timeout)

    def create_index_for_model(
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
        resp = self.index_api.create_index_for_model(req)

        if timeout == -1:
            return IndexModel(resp)
        return self.__poll_describe_index_until_ready(name, timeout)

    def __poll_describe_index_until_ready(self, name: str, timeout: Optional[int] = None):
        description = None

        def is_ready() -> bool:
            nonlocal description
            description = self.describe_index(name=name)
            return description.status.ready

        total_wait_time = 0
        if timeout is None:
            # Wait indefinitely
            while not is_ready():
                logger.debug(
                    f"Waiting for index {name} to be ready. Total wait time {total_wait_time} seconds."
                )
                total_wait_time += 5
                time.sleep(5)

        else:
            # Wait for a maximum of timeout seconds
            while not is_ready():
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
                time.sleep(5)
                timeout -= 5

        return description

    def delete_index(self, name: str, timeout: Optional[int] = None):
        self.index_api.delete_index(name)
        self.index_host_store.delete_host(self.config, name)

        if timeout == -1:
            return

        if timeout is None:
            while self.has_index(name):
                time.sleep(5)
        else:
            while self.has_index(name) and timeout >= 0:
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

    def list_indexes(self) -> IndexList:
        response = self.index_api.list_indexes()
        return IndexList(response)

    def describe_index(self, name: str) -> IndexModel:
        api_instance = self.index_api
        description = api_instance.describe_index(name)
        host = description.host
        self.index_host_store.set_host(self.config, name, host)

        return IndexModel(description)

    def has_index(self, name: str) -> bool:
        if name in self.list_indexes().names():
            return True
        else:
            return False

    def configure_index(
        self,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[Union[PodType, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        api_instance = self.index_api
        description = self.describe_index(name=name)

        req = PineconeDBControlRequestFactory.configure_index_request(
            description=description,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
        )
        api_instance.configure_index(name, configure_index_request=req)

    def create_collection(self, name: str, source: str) -> None:
        req = PineconeDBControlRequestFactory.create_collection_request(name=name, source=source)
        self.index_api.create_collection(create_collection_request=req)

    def list_collections(self) -> CollectionList:
        response = self.index_api.list_collections()
        return CollectionList(response)

    def delete_collection(self, name: str) -> None:
        self.index_api.delete_collection(name)

    def describe_collection(self, name: str):
        return self.index_api.describe_collection(name).to_dict()

    @staticmethod
    def from_texts(*args, **kwargs):
        """@private"""
        raise AttributeError(_build_langchain_attribute_error_message("from_texts"))

    @staticmethod
    def from_documents(*args, **kwargs):
        """@private"""
        raise AttributeError(_build_langchain_attribute_error_message("from_documents"))

    def Index(self, name: str = "", host: str = "", **kwargs):
        if name == "" and host == "":
            raise ValueError("Either name or host must be specified")

        pt = kwargs.pop("pool_threads", None) or self.pool_threads
        api_key = self.config.api_key
        openapi_config = self.openapi_config

        if host != "":
            check_realistic_host(host)

            # Use host url if it is provided
            index_host = normalize_host(host)
        else:
            # Otherwise, get host url from describe_index using the index name
            index_host = self.index_host_store.get_host(self.index_api, self.config, name)

        return _Index(
            host=index_host,
            api_key=api_key,
            pool_threads=pt,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs,
        )

    def IndexAsyncio(self, host: str, **kwargs):
        api_key = self.config.api_key
        openapi_config = self.openapi_config

        if host is None or host == "":
            raise ValueError("A host must be specified")

        check_realistic_host(host)
        index_host = normalize_host(host)

        return _IndexAsyncio(
            host=index_host,
            api_key=api_key,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs,
        )


def check_realistic_host(host: str) -> None:
    """@private

    Checks whether a user-provided host string seems plausible.
    Someone could erroneously pass an index name as the host by
    mistake, and if they have done that we'd like to give them a
    simple error message as feedback rather than attempting to
    call the url and getting a more cryptic DNS resolution error.
    """

    if "." not in host and "localhost" not in host:
        raise ValueError(
            f"You passed '{host}' as the host but this does not appear to be valid. Call describe_index() to confirm the host of the index."
        )
