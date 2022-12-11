# pinecone-client
The Pinecone python client

For more information, see the docs at https://www.pinecone.io/docs/

## Installation

Install a released version from pip:
```shell
pip install pinecone-client
```

Or the latest development version:
```shell
pip install git+https://git@github.com/pinecone-io/pinecone-python-client.git
```

Or a specific development version:
```shell
pip install git+https://git@github.com/pinecone-io/pinecone-python-client.git
pip install git+https://git@github.com/pinecone-io/pinecone-python-client.git@example-branch-name
pip install git+https://git@github.com/pinecone-io/pinecone-python-client.git@259deff
```

## Creating an index

The following example creates an index without a metadata
configuration. By default, Pinecone indexes all metadata.

```python

import pinecone


pinecone.init(api_key="YOUR_API_KEY",
              environment="us-west1-gcp")

pinecone.create_index("example-index", dimension=1024)
```

The following example creates an index that only indexes
the "color" metadata field. Queries against this index
cannot filter based on any other metadata field.

```python
metadata_config = {
    "indexed": ["color"]
}

pinecone.create_index("example-index-2", dimension=1024,
                      metadata_config=metadata_config)
```

## List indexes

The following example returns all indexes in your project.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

active_indexes = pinecone.list_indexes()
```

## Describe index

The following example returns information about the index `example-index`.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

index_description = pinecone.describe_index("example-index"
```)

## Delete an index

The following example deletes `example-index`.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

pinecone.delete_index("example-index")
```

## Scale replicas

The following example changes the number of replicas for `example-index`.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

new_number_of_replicas = 4
pinecone.configure_index("example-index", replicas=new_number_of_replicas)
```

## Describe index statistics

The following example returns statistics about the index `example-index`.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
index = pinecone.Index("example-index")

index_stats_response = index.describe_index_stats()
```

## Query an index

The following example queries the index `example-index` with metadata
filtering.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
index = pinecone.Index("example-index")

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
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
index = pinecone.Index("example-index")

delete_response = index.delete(ids=["vec1", "vec2"], namespace="example-namespace")
```

## Fetch vectors

The following example fetches vectors by ID.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
index = pinecone.Index("example-index")

fetch_response = index.fetch(ids=["vec1", "vec2"], namespace="example-namespace")
```


## Update vectors

The following example updates vectors by ID.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
index = pinecone.Index("example-index")

update_response = index.update(
    id="vec1",
    values=[0.1, 0.2, 0.3, 0.4],
    set_metadata={"genre": "drama"},
    namespace="example-namespace"
)
```

## Upsert vectors

The following example upserts vectors to `example-index`.

```python
import pinecone

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

## Create collection

The following example creates the collection `example-collection` from
`example-index`.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY",
              environment="us-west1-gcp")

pinecone.create_collection("example-collection", "example-index")
```

## List collections

The following example returns a list of the collections in the current project.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

active_collections = pinecone.list_collections()
```

## Describe a collection

The following example returns a description of the collection
`example-collection`.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

collection_description = pinecone.describe_collection("example-collection")
```

## Delete a collection

The following example deletes the collection `example-collection`.

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

pinecone.delete_collection("example-collection")
```


