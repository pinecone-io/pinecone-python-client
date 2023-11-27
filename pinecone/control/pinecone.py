import time
from typing import Optional, Dict, Any, Union, List, cast, NamedTuple

from .index_host_store import IndexHostStore

from pinecone.config import PineconeConfig, Config

from pinecone.core.client.api.manage_pod_indexes_api import ManagePodIndexesApi as IndexOperationsApi
from pinecone.core.client.api_client import ApiClient
from pinecone.utils import get_user_agent
from pinecone.core.client.models import (
    CreateCollectionRequest,
    CreateIndexRequest,
    ConfigureIndexRequest,
)
from pinecone.models import ServerlessSpec, PodSpec, IndexList, CollectionList

from pinecone.data import Index

class Pinecone:
    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        config: Optional[Config] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        index_api: Optional[IndexOperationsApi] = None,
        **kwargs,
    ):
        if config or kwargs.get("config"):
            configKwarg = config or kwargs.get("config")
            if not isinstance(configKwarg, Config):
                raise TypeError("config must be of type pinecone.config.Config")
            else:
                self.config = configKwarg
        else:
            self.config = PineconeConfig.build(api_key=api_key, host=host, additional_headers=additional_headers, **kwargs)

        if index_api:
            self.index_api = index_api
        else:
            api_client = ApiClient(configuration=self.config.openapi_config)
            api_client.user_agent = get_user_agent()
            extra_headers = self.config.additional_headers or {}
            for key, value in extra_headers.items():
                api_client.set_default_header(key, value)
            self.index_api = IndexOperationsApi(api_client)

        self.index_host_store = IndexHostStore()

    def create_index(
        self,
        name: str,
        dimension: int,
        spec: Union[Dict, ServerlessSpec, PodSpec],
        metric: Optional[str] = "cosine",
        timeout: Optional[int] = None,
    ):
        """Creates a Pinecone index.

        :param name: the name of the index.
        :type name: str
        :param dimension: the dimension of vectors that would be inserted in the index
        :type dimension: int
        :param metric: type of metric used in the vector index, one of `{"cosine", "dotproduct", "euclidean"}`, defaults to "cosine".
            Use "cosine" for cosine similarity,
            "dotproduct" for dot-product,
            and "euclidean" for euclidean distance.
        :type metric: str, optional
        :param spec: A dictionary containing configurations describing how the index should be deployed. For serverless indexes,
            specify region and cloud. For pod indexes, specify replicas, shards, pods, pod_type, metadata_config, and source_collection.
        :type spec: Dict
        :type timeout: int, optional
        :param timeout: Timeout for wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait. Default: None
        """

        api_instance = self.index_api

        if isinstance(spec, dict):
            api_instance.create_index(create_index_request=CreateIndexRequest(name=name, dimension=dimension, metric=metric, spec=spec))
        elif isinstance(spec, ServerlessSpec):
            api_instance.create_index(create_index_request=CreateIndexRequest(name=name, dimension=dimension, metric=metric, spec=spec.asdict()))
        elif isinstance(spec, PodSpec):
            api_instance.create_index(create_index_request=CreateIndexRequest(name=name, dimension=dimension, metric=metric, spec=spec.asdict()))

        def is_ready():
            status = self._get_status(name)
            ready = status["ready"]
            return ready

        if timeout == -1:
            return
        if timeout is None:
            while not is_ready():
                time.sleep(5)
        else:
            while (not is_ready()) and timeout >= 0:
                time.sleep(5)
                timeout -= 5
        if timeout and timeout < 0:
            raise (
                TimeoutError(
                    "Please call the describe_index API ({}) to confirm index status.".format(
                        "https://www.pinecone.io/docs/api/operation/describe_index/"
                    )
                )
            )

    def delete_index(self, name: str, timeout: Optional[int] = None):
        """Deletes a Pinecone index.

        :param name: the name of the index.
        :type name: str
        :param timeout: Timeout for wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait. Default: None
        :type timeout: int, optional
        """
        api_instance = self.index_api
        api_instance.delete_index(name)
        self.index_host_store.delete_host(self.config, name)

        def get_remaining():
            return name in api_instance.list_indexes()

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
        """Lists all indexes."""
        response = self.index_api.list_indexes()
        return IndexList(response)

    def describe_index(self, name: str):
        """Describes a Pinecone index.

        :param name: the name of the index to describe.
        :return: Returns an `IndexDescription` object
        """
        api_instance = self.index_api
        description = api_instance.describe_index(name)
        host = description.host
        self.index_host_store.set_host(self.config, name, host)

        return description

    def configure_index(self, name: str, replicas: Optional[int] = None, pod_type: Optional[str] = None):
        """Changes current configuration of the index.
        :param: name: the name of the Index
        :param: replicas: the desired number of replicas, lowest value is 0.
        :param: pod_type: the new pod_type for the index.
        """
        api_instance = self.index_api
        config_args: Dict[str, Any] = {}
        if pod_type:
            config_args.update(pod_type=pod_type)
        if replicas:
            config_args.update(replicas=replicas)
        configure_index_request = ConfigureIndexRequest(**config_args)
        api_instance.configure_index(name, configure_index_request=configure_index_request)

    def scale_index(self, name: str, replicas: int):
        """Change the number of replicas for the index. Replicas may be scaled up or down.

        :param name: the name of the Index
        :type name: str
        :param replicas: the number of replicas in the index now, lowest value is 1.
        :type replicas: int
        """
        api_instance = self.index_api
        api_instance.configure_index(name, patch_request=ConfigureIndexRequest(replicas=replicas, pod_type=""))

    def create_collection(self, name: str, source: str):
        """Create a collection
        :param name: Name of the collection
        :param source: Name of the source index
        """
        api_instance = self.index_api
        api_instance.create_collection(create_collection_request=CreateCollectionRequest(name=name, source=source))

    def list_collections(self) -> CollectionList:
        """List all collections"""
        api_instance = self.index_api
        response = api_instance.list_collections()
        return CollectionList(response)

    def delete_collection(self, name: str):
        """Deletes a collection.
        :param: name: The name of the collection
        """
        api_instance = self.index_api
        api_instance.delete_collection(name)

    def describe_collection(self, name: str):
        """Describes a collection.
        :param: The name of the collection
        :return: Description of the collection
        """
        api_instance = self.index_api
        return api_instance.describe_collection(name).to_dict()

    def _get_status(self, name: str):
        api_instance = self.index_api
        response = api_instance.describe_index(name)
        return response["status"]

    def Index(self, name: str):
        index_host = self.index_host_store.get_host(self.index_api, self.config, name)
        return Index(api_key=self.config.api_key, host=index_host)
