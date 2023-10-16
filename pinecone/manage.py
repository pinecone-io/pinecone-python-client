import time
from typing import List, NamedTuple, Optional

import pinecone
from pinecone.config import Config
from pinecone.core.client.api.index_operations_api import IndexOperationsApi
from pinecone.core.client.api_client import ApiClient
from pinecone.core.client.model.create_request import CreateRequest
from pinecone.core.client.model.patch_request import PatchRequest
from pinecone.core.client.model.create_collection_request import CreateCollectionRequest
from pinecone.core.utils import get_user_agent

__all__ = [
    "create_index",
    "delete_index",
    "describe_index",
    "list_indexes",
    "scale_index",
    "create_collection",
    "describe_collection",
    "list_collections",
    "delete_collection",
    "configure_index",
    "CollectionDescription",
    "IndexDescription",
]


class IndexDescription(NamedTuple):
    """
    Represents the description of an index.
    """
    name: str
    metric: str
    replicas: int
    dimension: int
    shards: int
    pods: int
    pod_type: str
    status: None
    metadata_config: None
    source_collection: None


class CollectionDescription(object):
    """
    Represents the description of a collection.
    """
    def __init__(self, keys, values):
        for k, v in zip(keys, values):
            self.__dict__[k] = v

    def __str__(self):
        return str(self.__dict__)


def _get_api_instance():
    client_config = Config.OPENAPI_CONFIG
    client_config.api_key = client_config.api_key or {}
    client_config.api_key["ApiKeyAuth"] = client_config.api_key.get("ApiKeyAuth", Config.API_KEY)
    client_config.server_variables = {**{"environment": Config.ENVIRONMENT}, **client_config.server_variables}
    api_client = ApiClient(configuration=client_config)
    api_client.user_agent = get_user_agent()
    api_instance = IndexOperationsApi(api_client)
    return api_instance


def _get_status(name: str):
    api_instance = _get_api_instance()
    response = api_instance.describe_index(name)
    return response["status"]


def create_index(
    name: str,
    dimension: int,
    timeout: int = None,
    index_type: str = "approximated",
    metric: str = "cosine",
    replicas: int = 1,
    shards: int = 1,
    pods: int = 1,
    pod_type: str = "p1",
    index_config: dict = None,
    metadata_config: dict = None,
    source_collection: str = "",
):
    """Creates a new index.

    Note that the index is not immediately ready to use. You can use the `timeout` parameter to control how long the ``create_index`` 
    call waits to return. You can use the ``describe_index`` function to check the status of an index.

    The minimum required configuration to create an index is the index name and dimension:

    ```python
    pinecone.create_index(name="my-index", dimension=128)
    ```

    In a more expansive example, you can specify the metric, number of pods, number of replicas, and pod type:

    ```python
    pinecone.create_index(
        name="my-index",
        dimension=1536,
        metric="cosine",
        pods=1,
        replicas=2,
        pod_type="p1.x1",
    )
    ``` 
    
    If you plan to begin upserting immediately after index creation is complete, you will want to leave `timeout` as the default `None`.
    In this case, the ``create_index`` call will block until the index is ready to use:

    ```python
    pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
    index = pinecone.Index("example-index")

    upsert_response = index.upsert(
        vectors=[
            ("vec1", [0.1, 0.2, 0.3, 0.4], {"genre": "drama"}),
            ("vec2", [0.2, 0.3, 0.4, 0.5], {"genre": "action"}),
        ],
        namespace="example-namespace"
    )
    ```

    Args:
        name (str): The name of the index. Must be unique within the project and contain only alphanumeric and hyphen characters.
            The name must start and end with alphanumeric characters.
        dimension (int): The dimension of the index. Must be a positive integer. The dimension of your index should match the 
            output dimension of your embedding model. For example, if you are using a model that outputs 128-dimensional vectors,
            you should set the dimension to 128.
        timeout (int, optional): Timeout in seconds to wait until an index is ready. If `None`, wait indefinitely until index is created; 
            if >=0, time out after this many seconds; if -1, return immediately and do not wait. Default: `None`
        index_type (str, optional): type of index, one of `{"approximated", "exact"}`, defaults to "approximated".
            The "approximated" index uses fast approximate search algorithms developed by Pinecone.
            The "exact" index uses accurate exact search algorithms.
            It performs exhaustive searches and thus it is usually slower than the "approximated" index.
        metric (str, optional): The metric specifies how similarity is calculated in the index when querying. The default
            metric is `'cosine'`. Supported metrics include `'cosine'`, `'dotproduct'`, and `'euclidean'`. To learn more
            about these options, see [Distance metrics](https://docs.pinecone.io/docs/indexes#distance-metrics).
        replicas (int, optional): The number of replicas in the index. The default number of replicas is 1. For more information
            see [Replicas](https://docs.pinecone.io/docs/manage-indexes/#replicas).
        shards (int, optional): The number of shards in the index. The default number of shards is 1.
        pods (int, optional): The number of pods in the index. The default number of pods is 1.
        pod_type (str, optional): The type of pod in the index. This string should combine a base pod type (`s1`, `p1`, or `p2`) with a
            size (`x1`, `x2`, `x4`, `x8`) into a string such as `p1.x1` or `s1.x4`. The default pod type is `p1.x1`. For more
            information on these, see this guide on [pod types and sizes](https://docs.pinecone.io/docs/indexes#pods-pod-types-and-pod-sizes).
        index_config (dict, optional): Advanced configuration options for the index
        metadata_config (dict, optional): Configuration for the behavior of Pinecone's internal metadata index. By default,
            all metadata is indexed; when a `metadata_config` is present, only metadata fields specified are indexed.
        source_collection (str, optional): If creating an index from a collection, you can specify the name of the collection here.
    """
    api_instance = _get_api_instance()

    api_instance.create_index(
        create_request=CreateRequest(
            name=name,
            dimension=dimension,
            index_type=index_type,
            metric=metric,
            replicas=replicas,
            shards=shards,
            pods=pods,
            pod_type=pod_type,
            index_config=index_config or {},
            metadata_config=metadata_config,
            source_collection=source_collection,
        )
    )

    def is_ready():
        status = _get_status(name)
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


def delete_index(name: str, timeout: int = None):
    """Deletes an index.

    Note that the index is not immediately deleted. You can use the `timeout` parameter to control how long the ``delete_index`` 
    call waits to return. You can use the ``list_indexes`` function to determine if an index has been deleted.
    
    Example:
    ```python
    pinecone.delete_index(name="my-index")
    ```

    Args:
        name (str): The name of the index to delete.
        timeout (int, optional): Timeout in seconds to wait until an index is deleted. If `None` wait indefinitely until index is deleted;
            if >=0, time out after this many seconds; if -1, return immediately and do not wait. Default: `None` 
    """
    api_instance = _get_api_instance()
    api_instance.delete_index(name)

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


def list_indexes():
    """Lists all Pinecone indexes.

    Example:
    ```python
    indexes = pinecone.list_indexes()
    print(indexes)
    # ["my-index", "my-other-index"]
    ```
    
    Returns:
        A list of index names.
    """
    api_instance = _get_api_instance()
    response = api_instance.list_indexes()
    return response


def describe_index(name: str):
    """Describe a Pinecone index.

    Example:
    ```python
    pinecone.describe_index("my-index")
    # {
    #   name='my-index', 
    #   metric='cosine',
    #   replicas=1,
    #   dimension=128,
    #   shards=1, 
    #   pods=1,
    #   pod_type='p1',
    #   status={'ready': True, 'state': 'RUNNING'},
    #   metadata_config=None, 
    #   source_collection=None
    # }
    ```

    Args:
        name(str): The name of the index to describe.
    
    Returns:
        An ``IndexDescription`` object.
    """
    api_instance = _get_api_instance()
    response = api_instance.describe_index(name)
    db = response["database"]
    ready = response["status"]["ready"]
    state = response["status"]["state"]
    return IndexDescription(
        name=db["name"],
        metric=db["metric"],
        replicas=db["replicas"],
        dimension=db["dimension"],
        shards=db["shards"],
        pods=db.get("pods", db["shards"] * db["replicas"]),
        pod_type=db.get("pod_type", "p1"),
        status={"ready": ready, "state": state},
        metadata_config=db.get("metadata_config"),
        source_collection=db.get("source_collection", ""),
    )


def scale_index(name: str, replicas: int):
    """Changes the number of replicas for the index, lowest value is 0.

    Example:
    ```python
    pinecone.scale_index(name="my-index", replicas=2)
    ```

    Args:
        name(str): The name of the index to scale.
        replicas(int): The new number of replicas for the index.
    """
    api_instance = _get_api_instance()
    api_instance.configure_index(name, patch_request=PatchRequest(replicas=replicas, pod_type=""))


def create_collection(name: str, source: str):
    """Create a new collection from an existing index.

    Example:
    ```python
    index_list = pinecone.list_indexes()
    pinecone.create_collection(name="my-collection", source=index_list[0])
    ```

    Args:
        name(str): The name of the collection you would like to create.
        source(str): The name of the index you would like to create the collection from.
    """
    api_instance = _get_api_instance()
    api_instance.create_collection(create_collection_request=CreateCollectionRequest(name=name, source=source))


def list_collections():
    """List all collections in a project.

    Example:
    ```python
    collection_list = pinecone.list_collections()
    print(collection_list)
    # ["my-collection", "my-other-collection"]
    ```
    
    Returns:
        A list of collection names.
    """
    api_instance = _get_api_instance()
    response = api_instance.list_collections()
    return response


def delete_collection(name: str):
    """Delete a collection by collection name.

    Example:
    ```python
    collection_list = pinecone.list_collections()
    collection_name = collection_list[0]
    pinecone.delete_collection(collection_name)
    ```

    Args:
        name(str): The name of the collection to delete.
    """
    api_instance = _get_api_instance()
    api_instance.delete_collection(name)


def describe_collection(name: str):
    """Describe a collection.

    Example:
    ```python
    pinecone.describe_collection("my-collection")
    # {
    #   'name': 'my-collection', 
    #   'status': 'Ready', 
    #   'size': 3089687, 
    #   'dimension': 3.0, 
    #   'vector_count': 2.0
    # }
    ```

    Args:
        name(str): The name of the collection to describe.
    
    Returns:
        A ``CollectionDescription`` object.
    """
    api_instance = _get_api_instance()
    response = api_instance.describe_collection(name).to_dict()
    response_object = CollectionDescription(response.keys(), response.values())
    return response_object


def configure_index(name: str, replicas: Optional[int] = None, pod_type: Optional[str] = ""):
    """Configure an index.

    Use this method to update configuration on an existing index. You can update the number of pods,
    replicas, and pod type.

    Example:
    ```python
    pinecone.configure_index(name="my-index", replicas=2, pod_type="p1.x2")

    ```
    Args:
        name(str): The name of the index to configure.
        replicas(int, optional): The desired number of replicas, lowest value is 0.
        pod_type(str, optional): The type of pod in the index. This string should combine a base pod type (`s1`, `p1`, or `p2`)
            with a size (`x1`, `x2`, `x4`, `x8`) into a string such as `p1.x1` or `s1.x4`. The default pod type is `p1.x1`,
            For more information on these, see this guide on [pod types and sizes](https://docs.pinecone.io/docs/indexes#pods-pod-types-and-pod-sizes).
    """
    api_instance = _get_api_instance()
    config_args = {}
    if pod_type != "":
        config_args.update(pod_type=pod_type)
    if replicas:
        config_args.update(replicas=replicas)
    patch_request = PatchRequest(**config_args)
    api_instance.configure_index(name, patch_request=patch_request)
