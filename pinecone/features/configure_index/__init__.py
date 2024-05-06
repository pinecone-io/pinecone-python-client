from typing import Optional, Dict, Any
import os
from pinecone.features.feature import PineconeFeature

from .generated import generated_configure_index

class ConfigureIndexFeature(PineconeFeature):
    def __init__(self):
        pass

    def location(self):
        return os.path.dirname(__file__)

    def target(self):
        return 'Pinecone'

    def methods(self):
        return {
            "configure_index": configure_index,
        }


def configure_index(self, name: str, replicas: Optional[int] = None, pod_type: Optional[str] = None):
        """This method is used to scale configuration fields for your pod-based Pinecone index. 

        :param: name: the name of the Index
        :param: replicas: the desired number of replicas, lowest value is 0.
        :param: pod_type: the new pod_type for the index. To learn more about the
            available pod types, please see [Understanding Indexes](https://docs.pinecone.io/docs/indexes)
        
        
        ```python
        from pinecone import Pinecone

        client = Pinecone()

        # Make a configuration change
        client.configure_index(name="my_index", replicas=4)

        # Call describe_index to see the index status as the 
        # change is applied.
        client.describe_index("my_index")
        ```

        """
        config_args: Dict[str, Any] = {}
        if pod_type:
            config_args.update(pod_type=pod_type)
        if replicas:
            config_args.update(replicas=replicas)

        generated_configure_index(self.http_client, name, spec={"pod": config_args})
