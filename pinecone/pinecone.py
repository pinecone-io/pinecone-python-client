import logging
from typing import Optional, Dict, Union, TYPE_CHECKING
from multiprocessing import cpu_count

from .legacy_pinecone_interface import LegacyPineconeDBControlInterface

from pinecone.config import PineconeConfig, ConfigBuilder

from pinecone.utils import normalize_host, PluginAware, docslinks
from .langchain_import_warnings import _build_langchain_attribute_error_message

logger = logging.getLogger(__name__)
""" @private """

if TYPE_CHECKING:
    from pinecone.db_data import (
        _Index as Index,
        _Inference as Inference,
        _IndexAsyncio as IndexAsyncio,
    )
    from pinecone.db_control import DBControl
    from pinecone.db_control.types import CreateIndexForModelEmbedTypedDict
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
        IndexModel,
        IndexList,
        CollectionList,
        IndexEmbed,
    )


class Pinecone(PluginAware, LegacyPineconeDBControlInterface):
    """
    A client for interacting with Pinecone APIs.
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

        self._inference: Optional["Inference"] = None  # Lazy initialization
        """ @private """

        self._db_control: Optional["DBControl"] = None  # Lazy initialization
        """ @private """

        super().__init__()  # Initialize PluginAware

    @property
    def inference(self) -> "Inference":
        """
        Inference is a namespace where an instance of the `pinecone.data.features.inference.inference.Inference` class is lazily created and cached.
        """
        if self._inference is None:
            from pinecone.db_data import _Inference

            self._inference = _Inference(config=self.config, openapi_config=self.openapi_config)
        return self._inference

    @property
    def db(self) -> "DBControl":
        """
        DBControl is a namespace where an instance of the `pinecone.control.db_control.DBControl` class is lazily created and cached.
        """
        if self._db_control is None:
            from pinecone.db_control import DBControl

            self._db_control = DBControl(
                config=self.config,
                openapi_config=self.openapi_config,
                pool_threads=self.pool_threads,
            )
        return self._db_control

    def create_index(
        self,
        name: str,
        spec: Union[Dict, "ServerlessSpec", "PodSpec"],
        dimension: Optional[int] = None,
        metric: Optional[Union["Metric", str]] = "cosine",
        timeout: Optional[int] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        vector_type: Optional[Union["VectorType", str]] = "dense",
        tags: Optional[Dict[str, str]] = None,
    ) -> "IndexModel":
        return self.db.index.create(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric,
            timeout=timeout,
            deletion_protection=deletion_protection,
            vector_type=vector_type,
            tags=tags,
        )

    def create_index_for_model(
        self,
        name: str,
        cloud: Union["CloudProvider", str],
        region: Union["AwsRegion", "GcpRegion", "AzureRegion", str],
        embed: Union["IndexEmbed", "CreateIndexForModelEmbedTypedDict"],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[
            Union["DeletionProtection", str]
        ] = "DeletionProtection.DISABLED",
        timeout: Optional[int] = None,
    ) -> "IndexModel":
        return self.db.index.create_for_model(
            name=name,
            cloud=cloud,
            region=region,
            embed=embed,
            tags=tags,
            deletion_protection=deletion_protection,
            timeout=timeout,
        )

    def delete_index(self, name: str, timeout: Optional[int] = None):
        return self.db.index.delete(name=name, timeout=timeout)

    def list_indexes(self) -> "IndexList":
        return self.db.index.list()

    def describe_index(self, name: str) -> "IndexModel":
        return self.db.index.describe(name=name)

    def has_index(self, name: str) -> bool:
        return self.db.index.has(name=name)

    def configure_index(
        self,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[Union["PodType", str]] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        return self.db.index.configure(
            name=name,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
        )

    def create_collection(self, name: str, source: str) -> None:
        return self.db.collection.create(name=name, source=source)

    def list_collections(self) -> "CollectionList":
        return self.db.collection.list()

    def delete_collection(self, name: str) -> None:
        return self.db.collection.delete(name=name)

    def describe_collection(self, name: str):
        return self.db.collection.describe(name=name)

    @staticmethod
    def from_texts(*args, **kwargs):
        """@private"""
        raise AttributeError(_build_langchain_attribute_error_message("from_texts"))

    @staticmethod
    def from_documents(*args, **kwargs):
        """@private"""
        raise AttributeError(_build_langchain_attribute_error_message("from_documents"))

    def Index(self, name: str = "", host: str = "", **kwargs) -> "Index":
        from pinecone.db_data import _Index

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
            index_host = self.db.index._get_host(name)

        return _Index(
            host=index_host,
            api_key=api_key,
            pool_threads=pt,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs,
        )

    def IndexAsyncio(self, host: str, **kwargs) -> "IndexAsyncio":
        from pinecone.db_data import _IndexAsyncio

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
