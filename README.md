# Pinecone Python Client &middot; ![License](https://img.shields.io/github/license/pinecone-io/pinecone-python-client?color=orange) [![CI](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/merge.yaml/badge.svg)](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/merge.yaml)


The official Pinecone Python client.

For more information, see the docs at https://www.pinecone.io/docs/

## Documentation

- If you are upgrading from a `2.2.x` version of the client, check out the [**v3 Migration Guide**](https://canyon-quilt-082.notion.site/Pinecone-Python-SDK-v3-0-0-Migration-Guide-056d3897d7634bf7be399676a4757c7b#a21aff70b403416ba352fd30e300bce3).
- [**Reference Documentation**](https://sdk.pinecone.io/python/index.html)


### Example code

Many of the brief examples shown in this README are using very small vectors to keep the documentation concise, but most real world usage will involve much larger embedding vectors. To see some more realistic examples of how this client can be used, explore some of our many Jupyter notebooks in the [examples](https://github.com/pinecone-io/examples) repository.


## Prerequisites

The Pinecone Python client is compatible with Python 3.8 and greater.


## Installation

There are two flavors of the Pinecone python client. The default client installed from PyPI as `pinecone-client` has a minimal set of dependencies and interacts with Pinecone via HTTP requests.

If you are aiming to maximimize performance, you can install additional gRPC dependencies to access an alternate client implementation that relies on gRPC for data operations. See the guide on [tuning performance](https://docs.pinecone.io/docs/performance-tuning).


### Installing with pip

```shell
# Install the latest version
pip3 install pinecone-client

# Install the latest version, with extra grpc dependencies
pip3 install "pinecone-client[grpc]"

# Install a specific version
pip3 install pinecone-client==3.0.0

# Install a specific version, with grpc extras
pip3 install "pinecone-client[grpc]"==3.0.0
```

### Installing with poetry

```shell
# Install the latest version
poetry add pinecone

# Install the latest version, with grpc extras
poetry add pinecone --extras grpc

# Install a specific version
poetry add pinecone-client==3.0.0

# Install a specific version, with grpc extras
poetry add pinecone-client==3.0.0 --extras grpc
```

## Usage

### Initializing the client

Before you can use the Pinecone SDK, you must sign up for an account and find your API key in the Pinecone console dashboard at [https://app.pinecone.io](https://app.pinecone.io).

#### Using environment variables

The `Pinecone` class is your main entry point into the Pinecone python SDK. If you have set your API Key in the `PINECONE_API_KEY` environment variable, you can instantiate the client with no other arguments.

```python
from pinecone import Pinecone

pc = Pinecone() # This reads the PINECONE_API_KEY env var
```

#### Using a configuration object

If you prefer to pass configuration in code, for example if you have a complex application that needs to interact with multiple different Pinecone projects, the constructor accepts a keyword argument for `api_key`. 

If you pass configuration in this way, you can have full control over what name to use for the environment variable, sidestepping any issues that would result 
from two different client instances both needing to read the same `PINECONE_API_KEY` variable that the client implicitly checks for.

Configuration passed with keyword arguments takes precedent over environment variables.

```python
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ.get('CUSTOM_VAR'))
```

### Working with GRPC (for improved performance)

If you've followed instructions above to install with optional `grpc` extras, you can unlock some performance improvements by working with an alternative version of the client imported from the `pinecone.grpc` subpackage.

```python
import os
from pinecone.grpc import PineconeGRPC

pc = PineconeGRPC(api_key=os.environ.get('PINECONE_API_KEY'))

# From here on, everything is identical to the REST-based client.
index = pc.Index(host='my-index-8833ca1.svc.us-east1-gcp.pinecone.io')

index.upsert(vectors=[])
index.query(vector=[...], top_key=10)
```

## Indexes

### Create Index

#### Create a serverless index

> [!WARNING]  
> Serverless indexes are in **public preview** and are available only on AWS in the
>  `us-west-2` region. Check the [current limitations](https://docs.pinecone.io/docs/limits#serverless-index-limitations) and test thoroughly before using it in production.

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
pc.create_index(
    name='my-index',
    dimension=1536,
    metric='euclidean',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-west-2'
    )
)
```

## Create a pod index

The following example creates an index without a metadata
configuration. By default, Pinecone indexes all metadata.

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
pc.create_index(
    name="example-index", 
    dimension=1536, 
    metric="cosine", 
    spec=PodSpec(
        environment='us-west-2', 
        pod_type='p1.x1'
    )
)
```

Pod indexes support many optional configuration fields. For example, 
the following example creates an index that only indexes
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
    spec=PodSpec(
        environment='us-west-2', 
        pod_type='p1.x1', 
        metadata_config=metadata_config
    )
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

The following example deletes the index named `example-index`.

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
import os
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
index = pc.Index(host=os.environ.get('INDEX_HOST'))

index_stats_response = index.describe_index_stats()
```


## Upsert vectors

The following example upserts vectors to `example-index`.

```python
import os
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
index = pc.Index(host=os.environ.get('INDEX_HOST'))

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
import os
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

# Find your index host by calling describe_index
# through the Pinecone web console
index = pc.Index(host=os.environ.get('INDEX_HOST'))

query_response = index.query(
    namespace="example-namespace",
    vector=[0.1, 0.2, 0.3, 0.4],
    top_k=10,
    include_values=True,
    include_metadata=True,
    filter={
        "genre": {"$in": ["comedy", "documentary", "drama"]}
    }
)
```

## Delete vectors

The following example deletes vectors by ID.

```python
import os
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

# Find your index host by calling describe_index
# through the Pinecone web console
index = pc.Index(host=os.environ.get('INDEX_HOST'))

delete_response = index.delete(ids=["vec1", "vec2"], namespace="example-namespace")
```

## Fetch vectors

The following example fetches vectors by ID.

```python
import os
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

# Find your index host by calling describe_index
# through the Pinecone web console
index = pc.Index(host=os.environ.get('INDEX_HOST'))

fetch_response = index.fetch(ids=["vec1", "vec2"], namespace="example-namespace")
```

## Update vectors

The following example updates vectors by ID.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

# Find your index host by calling describe_index
# through the Pinecone web console
index = pc.Index(host=os.environ.get('INDEX_HOST'))

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

pc.create_collection(
    name="example-collection", 
    source="example-index"
)
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
