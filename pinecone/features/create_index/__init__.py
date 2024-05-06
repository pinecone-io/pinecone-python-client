import time
from typing import Optional, Dict, Union
from pinecone.models import ServerlessSpec, PodSpec
import os

from pinecone.features.feature import PineconeFeature

from .generated import generated_create_index

class CreateIndexFeature(PineconeFeature):
    def __init__(self):
        pass

    def location(self):
        return os.path.dirname(__file__)

    def target(self):
        return 'Pinecone'

    def methods(self):
        return {
            "create_index": create_index,
        }

def create_index(
    self,
    name: str,
    dimension: int,
    spec: Union[Dict, ServerlessSpec, PodSpec],
    metric: Optional[str] = "cosine",
    timeout: Optional[int] = None,
):
    """Creates a Pinecone index.

    :param name: The name of the index to create. Must be unique within your project and 
        cannot be changed once created. Allowed characters are lowercase letters, numbers, 
        and hyphens and the name may not begin or end with hyphens. Maximum length is 45 characters.
    :type name: str
    :param dimension: The dimension of vectors that will be inserted in the index. This should
        match the dimension of the embeddings you will be inserting. For example, if you are using
        OpenAI's CLIP model, you should use `dimension=1536`.
    :type dimension: int
    :param metric: Type of metric used in the vector index when querying, one of `{"cosine", "dotproduct", "euclidean"}`. Defaults to `"cosine"`.
        Defaults to `"cosine"`.
    :type metric: str, optional
    :param spec: A dictionary containing configurations describing how the index should be deployed. For serverless indexes,
        specify region and cloud. For pod indexes, specify replicas, shards, pods, pod_type, metadata_config, and source_collection.
    :type spec: Dict
    :type timeout: int, optional
    :param timeout: Specify the number of seconds to wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds;
        if -1, return immediately and do not wait. Default: None

    ### Creating a serverless index
    
    ```python
    import os
    from pinecone import Pinecone, ServerlessSpec

    client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    client.create_index(
        name="my_index", 
        dimension=1536, 
        metric="cosine", 
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
    ```

    ### Creating a pod index

    ```python
    import os
    from pinecone import Pinecone, PodSpec

    client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    client.create_index(
        name="my_index",
        dimension=1536,
        metric="cosine",
        spec=PodSpec(
            environment="us-east1-gcp", 
            pod_type="p1.x1"
        )
    )
    ```
    """

    if isinstance(spec, dict):
        generated_create_index(self.http_client, name=name, dimension=dimension, metric=metric, spec=spec)
    elif isinstance(spec, ServerlessSpec):
        generated_create_index(self.http_client, name=name, dimension=dimension, metric=metric, spec=spec.asdict())
    elif isinstance(spec, PodSpec):
        generated_create_index(self.http_client, name=name, dimension=dimension, metric=metric, spec=spec.asdict())
    else:
        raise TypeError("spec must be of type dict, ServerlessSpec, or PodSpec")

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