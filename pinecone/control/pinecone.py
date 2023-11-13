import time
from typing import Optional, Dict, Any, cast

from .index_host_store import IndexHostStore

from pinecone.config import PineconeConfig, Config


from pinecone.core.client.api.index_operations_api import IndexOperationsApi
from pinecone.core.client.api_client import ApiClient
from pinecone.core.client.models import CreateCollectionRequest, CreateRequest, PatchRequest
from pinecone.utils import get_user_agent
from pinecone.models import ListIndexesResponse, ListIndexMeta, IndexDescription, IndexStatus, CollectionDescription

from pinecone.data import Index

class Pinecone:
    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        config: Optional[Config] = None,
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
            self.config = PineconeConfig.build(api_key=api_key, host=host, **kwargs)
        
        if index_api:
            self.index_api = index_api
        else:
            api_client = ApiClient(configuration=self.config.openapi_config)
            api_client.user_agent = get_user_agent()
            self.index_api = IndexOperationsApi(api_client)

        self.index_host_store = IndexHostStore()

    def create_index(
        self,
        name: str,
        dimension: int,
        cloud: str,
        region: str,
        capacity_mode: str,
        environment: Optional[str] = None,
        timeout: Optional[int] = None,
        index_type: str = "approximated",
        metric: str = "cosine",
        replicas: Optional[int] = None,
        shards: Optional[int] = None,
        pods: Optional[int] = None,
        pod_type: Optional[str] = None,
        metadata_config: Optional[dict] = None,
        source_collection: str = "",
    ):
        """Creates a Pinecone index.

        :param name: the name of the index.
        :type name: str
        :param dimension: the dimension of vectors that would be inserted in the index
        :type dimension: int
        :param cloud: The cloud where you would like your index hosted. One of `{"aws", "gcp"}`.
        :type cloud: str
        :param region: The region where you would like your index hosted.
        :type region: str
        :param capacity_mode: The capacity mode for the index. One of `{"pod"}`.
        :type capacity_mode: str
        :param environment: The environment where you would like your index hosted, e.g. 'us-east1-gcp'. Find this value in the Pinecone web console along with your API keys.
        :type environment: str, optional
        :param index_type: type of index, one of `{"approximated", "exact"}`, defaults to "approximated".
            The "approximated" index uses fast approximate search algorithms developed by Pinecone.
            The "exact" index uses accurate exact search algorithms.
            It performs exhaustive searches and thus it is usually slower than the "approximated" index.
        :type index_type: str, optional
        :param metric: type of metric used in the vector index, one of `{"cosine", "dotproduct", "euclidean"}`, defaults to "cosine".
            Use "cosine" for cosine similarity,
            "dotproduct" for dot-product,
            and "euclidean" for euclidean distance.
        :type metric: str, optional
        :param replicas: the number of replicas, defaults to 1.
            Use at least 2 replicas if you need high availability (99.99% uptime) for querying.
            For additional throughput (QPS) your index needs to support, provision additional replicas.
        :type replicas: int, optional
        :param shards: the number of shards per index, defaults to 1.
            Use 1 shard per 1GB of vectors
        :type shards: int,optional
        :param pods: Total number of pods to be used by the index. pods = shard*replicas
        :type pods: int,optional
        :param pod_type: the pod type to be used for the index. can be one of p1 or s1.
        :type pod_type: str,optional
        :param index_config: Advanced configuration options for the index
        :param metadata_config: Configuration related to the metadata index
        :type metadata_config: dict, optional
        :param source_collection: Collection name to create the index from
        :type metadata_config: str, optional
        :type timeout: int, optional
        :param timeout: Timeout for wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait. Default: None
        """
        create_args = locals()
        del create_args["self"]
        if environment is None:
            del create_args["environment"]

        api_instance = self.index_api
        api_instance.create_index(create_request=CreateRequest(**create_args))

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

    def list_indexes(self) -> ListIndexesResponse:
        """Lists all indexes."""
        response = self.index_api.list_indexes()
        return ListIndexesResponse(databases=[ListIndexMeta(**index.to_dict()) for index in response.databases])

    def describe_index(self, name: str) -> IndexDescription:
        """Describes a Pinecone index.

        :param name: the name of the index to describe.
        :return: Returns an `IndexDescription` object
        """
        api_instance = self.index_api
        response = api_instance.describe_index(name)
        db = response.database
        host = response.status.host

        self.index_host_store.set_host(self.config, name, host)

        return IndexDescription(
            name=db.name,
            metric=db.metric,
            replicas=db.replicas,
            dimension=db.dimension,
            shards=db.shards,
            pods=db.pods,
            capacity_mode=db.capacity_mode,
            pod_type=db.pod_type,
            status=IndexStatus(ready=response.status.ready, state=response.status.state, host=response.status.host),
            metadata_config=db.metadata_config,
        )

    def configure_index(self, name: str, replicas: Optional[int] = None, pod_type: Optional[str] = ""):
        """Changes current configuration of the index.
        :param: name: the name of the Index
        :param: replicas: the desired number of replicas, lowest value is 0.
        :param: pod_type: the new pod_type for the index.
        """
        api_instance = self.index_api
        config_args: Dict[str, Any] = {}
        if pod_type != "":
            config_args.update(pod_type=pod_type)
        if replicas:
            config_args.update(replicas=replicas)
        patch_request = PatchRequest(**config_args)
        api_instance.configure_index(name, patch_request=patch_request)

    def scale_index(self, name: str, replicas: int):
        """Increases number of replicas for the index.

        :param name: the name of the Index
        :type name: str
        :param replicas: the number of replicas in the index now, lowest value is 0.
        :type replicas: int
        """
        api_instance = self.index_api
        api_instance.configure_index(name, patch_request=PatchRequest(replicas=replicas, pod_type=""))

    def create_collection(self, name: str, source: str):
        """Create a collection
        :param name: Name of the collection
        :param source: Name of the source index
        """
        api_instance = self.index_api
        api_instance.create_collection(create_collection_request=CreateCollectionRequest(name=name, source=source))

    def list_collections(self):
        """List all collections"""
        api_instance = self.index_api
        response = api_instance.list_collections()
        return response

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
        response = api_instance.describe_collection(name).to_dict()
        response_object = CollectionDescription(response.keys(), response.values())
        return response_object

    def _get_status(self, name: str):
        api_instance = self.index_api
        response = api_instance.describe_index(name)
        return response["status"]

    def Index(self, name: str):
        index_host = self.index_host_store.get_host(self.index_api, self.config, name)
        return Index(api_key=self.config.api_key, host=index_host)
