from pinecone.features.feature import PineconeFeature
from pinecone.models.index_list import IndexList
import os

from .generated import generated_list_indexes

class ListIndexFeature(PineconeFeature):
    def __init__(self):
        pass

    def location(self):
        return os.path.dirname(__file__)

    def target(self):
        return 'Pinecone'

    def methods(self):
        return {
            "list_indexes": list_indexes,
        }

def list_indexes(self) -> IndexList:
        """Lists all indexes.
        
        The results include a description of all indexes in your project, including the 
        index name, dimension, metric, status, and spec.

        :return: Returns an `IndexList` object, which is iterable and contains a 
            list of `IndexDescription` objects. It also has a convenience method `names()`
            which returns a list of index names.

        ```python
        from pinecone import Pinecone

        client = Pinecone()

        index_name = "my_index"
        if index_name not in client.list_indexes().names():
            print("Index does not exist, creating...")
            client.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
        ```
        
        You can also use the `list_indexes()` method to iterate over all indexes in your project
        and get other information besides just names.

        ```python
        from pinecone import Pinecone

        client = Pinecone()

        for index in client.list_indexes():
            print(index.name)
            print(index.dimension)
            print(index.metric)
            print(index.status)
            print(index.host)
            print(index.spec)
        ```

        """
        return generated_list_indexes(self.http_client)