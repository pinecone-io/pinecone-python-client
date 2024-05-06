import os
from pinecone.features.feature import PineconeFeature

from .generated import generated_describe_index

class DescribeIndexFeature(PineconeFeature):
    def __init__(self):
        pass

    def location(self):
        return os.path.dirname(__file__)

    def target(self):
        return 'Pinecone'

    def methods(self):
        return {
            "describe_index": describe_index,
        }


def describe_index(self, name: str):
    """Describes a Pinecone index.

    :param name: the name of the index to describe.
    :return: Returns an `IndexDescription` object
    which gives access to properties such as the 
    index name, dimension, metric, host url, status, 
    and spec.

    ### Getting your index host url

    In a real production situation, you probably want to
    store the host url in an environment variable so you
    don't have to call describe_index and re-fetch it 
    every time you want to use the index. But this example
    shows how to get the value from the API using describe_index.

    ```python
    from pinecone import Pinecone, Index

    client = Pinecone()

    description = client.describe_index("my_index")
    
    host = description.host
    print(f"Your index is hosted at {description.host}")

    index = client.Index(name="my_index", host=host)
    index.upsert(vectors=[...])
    ```
    """
    description = generated_describe_index(self.http_client, name)
    host = description.host
    self.index_host_store.set_host(self.config, name, host)

    return description