# pinecone-client
The Pinecone python client

For more information, see the docs at https://www.pinecone.io/docs/

## Installation

Install a released version from pip:
```shell
pip3 install pinecone-client
```

Or the gRPC version of the client for [tuning performance](https://docs.pinecone.io/docs/performance-tuning)

```shell
pip3 install "pinecone-client[grpc]"
```

Or the latest development version:
```shell
pip3 install git+https://git@github.com/pinecone-io/pinecone-python-client.git
```

Or a specific development version:
```shell
pip3 install git+https://git@github.com/pinecone-io/pinecone-python-client.git
pip3 install git+https://git@github.com/pinecone-io/pinecone-python-client.git@example-branch-name
pip3 install git+https://git@github.com/pinecone-io/pinecone-python-client.git@259deff
```

## Create a client
The following example creates an authenticated client.

```python
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
```
To use gRPC for interacting with an index, instantiate with `PineconeGRPC` 
```python
import os
from pinecone.grpc import PineconeGRPC

pc = PineconeGRPC(api_key=os.environ.get('PINECONE_API_KEY'))
# Functionality matches non-gRPC client
index = pc.Index('index-name')
```

## Create an index

The following example creates an index without a metadata
configuration. By default, Pinecone indexes all metadata.

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
pc.create_index("example-index", dimension=1536, metric="cosine", spec=PodSpec(environment='us-west-2', pod_type='p1.x1'))

```

The following example creates an index that only indexes
the "color" metadata field. Queries against this index
cannot filter based on any other metadata field.

```python
from pinecone import Pinecone, PodSpec
pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

metadata_config = {
    "indexed": ["color"]
}

pc.create_index(
    "example-index-2",
    dimension=1536,
    spec=PodSpec(environment='us-west-2', pod_type='p1.x1', metadata_config=metadata_config)
)
```

## List indexes

The following example returns all indexes in your project.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
for index in pc.list_indexes():
    print(index['name'])
```

## Describe index

The following example returns information about the index `example-index`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

index_description = pc.describe_index("example-index")
```

## Delete an index

The following example deletes `example-index`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.delete_index("example-index")
```

## Scale replicas

The following example changes the number of replicas for `example-index`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

new_number_of_replicas = 4
pc.configure_index("example-index", replicas=new_number_of_replicas)
```

## Describe index statistics

The following example returns statistics about the index `example-index`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
index = pc.Index("example-index")

index_stats_response = index.describe_index_stats()
```


## Upsert vectors

The following example upserts vectors to `example-index`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
index = pc.Index("example-index")

upsert_response = index.upsert(
    vectors=[
        ("vec1", [0.1, 0.2, 0.3, 0.4], {"genre": "drama"}),
        ("vec2", [0.2, 0.3, 0.4, 0.5], {"genre": "action"}),
    ],
    namespace="example-namespace"
)
```

## Query an index

The following example queries the index `example-index` with metadata
filtering.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
index = pc.Index("example-index")

query_response = index.query(
    namespace="example-namespace",
    top_k=10,
    include_values=True,
    include_metadata=True,
    vector=[0.1, 0.2, 0.3, 0.4],
    filter={
        "genre": {"$in": ["comedy", "documentary", "drama"]}
    }
)
```

## Delete vectors

The following example deletes vectors by ID.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
index = pc.Index("example-index")

delete_response = index.delete(ids=["vec1", "vec2"], namespace="example-namespace")
```

## Fetch vectors

The following example fetches vectors by ID.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
index = pc.Index("example-index")

fetch_response = index.fetch(ids=["vec1", "vec2"], namespace="example-namespace")
```

## Update vectors

The following example updates vectors by ID.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
index = pc.Index("example-index")

update_response = index.update(
    id="vec1",
    values=[0.1, 0.2, 0.3, 0.4],
    set_metadata={"genre": "drama"},
    namespace="example-namespace"
)
```

## Create collection

The following example creates the collection `example-collection` from
`example-index`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_collection("example-collection", "example-index")
```

## List collections

The following example returns a list of the collections in the current project.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

active_collections = pc.list_collections()
```

## Describe a collection

The following example returns a description of the collection
`example-collection`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

collection_description = pc.describe_collection("example-collection")
```

## Delete a collection

The following example deletes the collection `example-collection`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.delete_collection("example-collection")
```

# Contributing 

If you'd like to make a contribution, or get setup locally to develop the Pinecone python client, please see our [contributing guide](./CONTRIBUTING.md)
