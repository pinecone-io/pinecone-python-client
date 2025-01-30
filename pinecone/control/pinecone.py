import time
import logging
from typing import Optional, Dict, Any, Union
from enum import Enum

from .index_host_store import IndexHostStore
from .pinecone_interface import PineconeDBControlInterface

from pinecone.config import PineconeConfig, Config, ConfigBuilder

from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
from pinecone.openapi_support.api_client import ApiClient


from pinecone.utils import (
    normalize_host,
    setup_openapi_client,
    build_plugin_setup_client,
    convert_enum_to_string,
)
from pinecone.core.openapi.db_control.models import (
    CreateCollectionRequest,
    CreateIndexForModelRequest,
    CreateIndexForModelRequestEmbed,
    CreateIndexRequest,
    ConfigureIndexRequest,
    ConfigureIndexRequestSpec,
    ConfigureIndexRequestSpecPod,
    DeletionProtection as DeletionProtectionModel,
    IndexSpec,
    IndexTags,
    ServerlessSpec as ServerlessSpecModel,
    PodSpec as PodSpecModel,
    PodSpecMetadataConfig,
)
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
from pinecone.utils import parse_non_empty_args, docslinks

from pinecone.data import _Index, _AsyncioIndex, _Inference
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

from pinecone_plugin_interface import load_and_install as install_plugins

logger = logging.getLogger(__name__)
""" @private """


class Pinecone(PineconeDBControlInterface):
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
        config: Optional[Config] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        pool_threads: Optional[int] = 1,
        index_api: Optional[ManageIndexesApi] = None,
        **kwargs,
    ):
        if config:
            if not isinstance(config, Config):
                raise TypeError("config must be of type pinecone.config.Config")
            else:
                self.config = config
        else:
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

        if kwargs.get("openapi_config", None):
            raise Exception(
                "Passing openapi_config is no longer supported. Please pass settings such as proxy_url, proxy_headers, ssl_ca_certs, and ssl_verify directly to the Pinecone constructor as keyword arguments. See the README at https://github.com/pinecone-io/pinecone-python-client for examples."
            )

        self.openapi_config = ConfigBuilder.build_openapi_config(self.config, **kwargs)
        self.pool_threads = pool_threads

        self._inference = None  # Lazy initialization

        if index_api:
            self.index_api = index_api
        else:
            self.index_api = setup_openapi_client(
                api_client_klass=ApiClient,
                api_klass=ManageIndexesApi,
                config=self.config,
                openapi_config=self.openapi_config,
                pool_threads=pool_threads,
                api_version=API_VERSION,
            )

        self.index_host_store = IndexHostStore()
        """ @private """

        self.load_plugins()

    @property
    def inference(self):
        """Dynamically create and cache the Inference instance."""
        if self._inference is None:
            self._inference = _Inference(config=self.config, openapi_config=self.openapi_config)
        return self._inference

    def load_plugins(self):
        """@private"""
        try:
            # I don't expect this to ever throw, but wrapping this in a
            # try block just in case to make sure a bad plugin doesn't
            # halt client initialization.
            openapi_client_builder = build_plugin_setup_client(
                config=self.config,
                openapi_config=self.openapi_config,
                pool_threads=self.pool_threads,
            )
            install_plugins(self, openapi_client_builder)
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")

    def __parse_tags(self, tags: Optional[Dict[str, str]]) -> IndexTags:
        if tags is None:
            return IndexTags()
        else:
            return IndexTags(**tags)

    def __parse_deletion_protection(
        self, deletion_protection: Union[DeletionProtection, str]
    ) -> DeletionProtectionModel:
        deletion_protection = convert_enum_to_string(deletion_protection)
        if deletion_protection in ["enabled", "disabled"]:
            return DeletionProtectionModel(deletion_protection)
        else:
            raise ValueError("deletion_protection must be either 'enabled' or 'disabled'")

    def __parse_index_spec(self, spec: Union[Dict, ServerlessSpec, PodSpec]) -> IndexSpec:
        if isinstance(spec, dict):
            if "serverless" in spec:
                index_spec = IndexSpec(serverless=ServerlessSpecModel(**spec["serverless"]))
            elif "pod" in spec:
                args_dict = parse_non_empty_args(
                    [
                        ("environment", spec["pod"].get("environment")),
                        ("metadata_config", spec["pod"].get("metadata_config")),
                        ("replicas", spec["pod"].get("replicas")),
                        ("shards", spec["pod"].get("shards")),
                        ("pods", spec["pod"].get("pods")),
                        ("source_collection", spec["pod"].get("source_collection")),
                    ]
                )
                if args_dict.get("metadata_config"):
                    args_dict["metadata_config"] = PodSpecMetadataConfig(
                        indexed=args_dict["metadata_config"].get("indexed", None)
                    )
                index_spec = IndexSpec(pod=PodSpecModel(**args_dict))
            else:
                raise ValueError("spec must contain either 'serverless' or 'pod' key")
        elif isinstance(spec, ServerlessSpec):
            index_spec = IndexSpec(
                serverless=ServerlessSpecModel(cloud=spec.cloud, region=spec.region)
            )
        elif isinstance(spec, PodSpec):
            args_dict = parse_non_empty_args(
                [
                    ("replicas", spec.replicas),
                    ("shards", spec.shards),
                    ("pods", spec.pods),
                    ("source_collection", spec.source_collection),
                ]
            )
            if spec.metadata_config:
                args_dict["metadata_config"] = PodSpecMetadataConfig(
                    indexed=spec.metadata_config.get("indexed", None)
                )

            index_spec = IndexSpec(
                pod=PodSpecModel(environment=spec.environment, pod_type=spec.pod_type, **args_dict)
            )
        else:
            raise TypeError("spec must be of type dict, ServerlessSpec, or PodSpec")

        return index_spec

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
        if metric is not None:
            metric = convert_enum_to_string(metric)
        if vector_type is not None:
            vector_type = convert_enum_to_string(vector_type)
        if deletion_protection is not None:
            dp = self.__parse_deletion_protection(deletion_protection)
        else:
            dp = None

        tags_obj = self.__parse_tags(tags)
        index_spec = self.__parse_index_spec(spec)

        if vector_type == VectorType.SPARSE.value and dimension is not None:
            raise ValueError("dimension should not be specified for sparse indexes")

        args = parse_non_empty_args(
            [
                ("name", name),
                ("dimension", dimension),
                ("metric", metric),
                ("spec", index_spec),
                ("deletion_protection", dp),
                ("vector_type", vector_type),
                ("tags", tags_obj),
            ]
        )

        req = CreateIndexRequest(**args)
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
        cloud = convert_enum_to_string(cloud)
        region = convert_enum_to_string(region)
        if deletion_protection is not None:
            dp = self.__parse_deletion_protection(deletion_protection)
        else:
            dp = None
        tags_obj = self.__parse_tags(tags)

        if isinstance(embed, IndexEmbed):
            parsed_embed = embed.as_dict()
        else:
            # if dict, we need to parse enum values, if any, to string
            # and verify required fields are present
            required_fields = ["model", "field_map"]
            for field in required_fields:
                if field not in embed:
                    raise ValueError(f"{field} is required in embed")
            parsed_embed = {}
            for key, value in embed.items():
                if isinstance(value, Enum):
                    parsed_embed[key] = convert_enum_to_string(value)
                else:
                    parsed_embed[key] = value

        args = parse_non_empty_args(
            [
                ("name", name),
                ("cloud", cloud),
                ("region", region),
                ("embed", CreateIndexForModelRequestEmbed(**parsed_embed)),
                ("deletion_protection", dp),
                ("tags", tags_obj),
            ]
        )

        req = CreateIndexForModelRequest(**args)
        resp = self.index_api.create_index_for_model(req)

        if timeout == -1:
            return IndexModel(resp)
        return self.__poll_describe_index_until_ready(name, timeout)

    def __poll_describe_index_until_ready(self, name: str, timeout: Optional[int] = None):
        description = None

        def is_ready():
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

        def get_remaining():
            return name in self.list_indexes().names()

        if timeout == -1:
            return

        if timeout is None:
            while get_remaining():
                time.sleep(5)
        else:
            while get_remaining() and timeout >= 0:
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

        if deletion_protection is None:
            dp = DeletionProtectionModel(description.deletion_protection)
        elif isinstance(deletion_protection, DeletionProtection):
            dp = DeletionProtectionModel(deletion_protection.value)
        elif deletion_protection in ["enabled", "disabled"]:
            dp = DeletionProtectionModel(deletion_protection)
        else:
            raise ValueError("deletion_protection must be either 'enabled' or 'disabled'")

        fetched_tags = description.tags
        if fetched_tags is None:
            starting_tags = {}
        else:
            starting_tags = fetched_tags.to_dict()

        if tags is None:
            # Do not modify tags if none are provided
            tags = starting_tags
        else:
            # Merge existing tags with new tags
            tags = {**starting_tags, **tags}

        pod_config_args: Dict[str, Any] = {}
        if pod_type:
            new_pod_type = pod_type.value if isinstance(pod_type, PodType) else pod_type
            pod_config_args.update(pod_type=new_pod_type)
        if replicas:
            pod_config_args.update(replicas=replicas)

        if pod_config_args != {}:
            spec = ConfigureIndexRequestSpec(pod=ConfigureIndexRequestSpecPod(**pod_config_args))
            req = ConfigureIndexRequest(deletion_protection=dp, spec=spec, tags=IndexTags(**tags))
        else:
            req = ConfigureIndexRequest(deletion_protection=dp, tags=IndexTags(**tags))

        api_instance.configure_index(name, configure_index_request=req)

    def create_collection(self, name: str, source: str):
        api_instance = self.index_api
        api_instance.create_collection(
            create_collection_request=CreateCollectionRequest(name=name, source=source)
        )

    def list_collections(self) -> CollectionList:
        api_instance = self.index_api
        response = api_instance.list_collections()
        return CollectionList(response)

    def delete_collection(self, name: str):
        api_instance = self.index_api
        api_instance.delete_collection(name)

    def describe_collection(self, name: str):
        api_instance = self.index_api
        return api_instance.describe_collection(name).to_dict()

    @staticmethod
    def from_texts(*args, **kwargs):
        raise AttributeError(_build_langchain_attribute_error_message("from_texts"))

    @staticmethod
    def from_documents(*args, **kwargs):
        raise AttributeError(_build_langchain_attribute_error_message("from_documents"))

    def Index(self, name: str = "", host: str = "", **kwargs):
        if name == "" and host == "":
            raise ValueError("Either name or host must be specified")

        pt = kwargs.pop("pool_threads", None) or self.pool_threads
        api_key = self.config.api_key
        openapi_config = self.openapi_config

        if host != "":
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

    def AsyncioIndex(self, name: str = "", host: str = "", **kwargs):
        if name == "" and host == "":
            raise ValueError("Either name or host must be specified")

        pt = kwargs.pop("pool_threads", None) or self.pool_threads
        api_key = self.config.api_key
        openapi_config = self.openapi_config

        if host != "":
            # Use host url if it is provided
            index_host = normalize_host(host)
        else:
            # Otherwise, get host url from describe_index using the index name
            index_host = self.index_host_store.get_host(self.index_api, self.config, name)

        return _AsyncioIndex(
            host=index_host,
            api_key=api_key,
            pool_threads=pt,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs,
        )
