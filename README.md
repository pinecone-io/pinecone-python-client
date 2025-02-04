# Pinecone Python SDK
![License](https://img.shields.io/github/license/pinecone-io/pinecone-python-client?color=orange) [![CI](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/pr.yaml/badge.svg)](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/pr.yaml)

The official Pinecone Python SDK.

For more information, see the docs at https://docs.pinecone.io


## Documentation

- [**Reference Documentation**](https://sdk.pinecone.io/python/index.html)

### Upgrading the SDK



### Example code

Many of the brief examples shown in this README are using very small vectors to keep the documentation concise, but most real world usage will involve much larger embedding vectors. To see some more realistic examples of how this SDK can be used, explore some of our many Jupyter notebooks in the [examples](https://github.com/pinecone-io/examples) repository.

## Prerequisites

The Pinecone Python SDK is compatible with Python 3.9 and greater. It has been tested with CPython versions from 3.9 to 3.13.

## Installation

The Pinecone Python SDK is distributed on PyPI using the package name `pinecone`. By default the `pinecone` has a minimal set of dependencies, but you can install some extras to unlock additional functionality.

Available extras:

- `pinecone[asyncio]` will add a dependency on `aiohttp` and enable usage of `PineconeAsyncio`, the asyncio-enabled version of the client for use with highly asynchronous modern web frameworks such as FastAPI.
- `pinecone[grpc]` will add dependencies on `grpcio` and related libraries needed to make pinecone data calls such as `upsert` and `query` over [GRPC](https://grpc.io/) for a modest performance improvement. See the guide on [tuning performance](https://docs.pinecone.io/docs/performance-tuning).

#### Installing with pip

```shell
# Install the latest version
pip3 install pinecone

# Install the latest version, with optional dependencies
pip3 install "pinecone[asyncio,grpc]"
```

#### Installing with uv

[uv](https://docs.astral.sh/uv/) is a modern package manager that runs 10-100x faster than pip and supports most pip syntax.

```shell
# Install the latest version
uv install pinecone

# Install the latest version, optional dependencies
uv install "pinecone[asyncio,grpc]"
```

#### Installing with [poetry](https://python-poetry.org/)

```shell
# Install the latest version
poetry add pinecone

# Install the latest version, with optional dependencies
poetry add pinecone --extras asyncio --extras grpc
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

#### Using configuration keyword params

If you prefer to pass configuration in code, for example if you have a complex application that needs to interact with multiple different Pinecone projects, the constructor accepts a keyword argument for `api_key`.

If you pass configuration in this way, you can have full control over what name to use for the environment variable, sidestepping any issues that would result
from two different client instances both needing to read the same `PINECONE_API_KEY` variable that the client implicitly checks for.

Configuration passed with keyword arguments takes precedence over environment variables.

```python
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ.get('CUSTOM_VAR'))
```

# Indexes

## Create Index

### Create a serverless index


#### Create a serverless index on Amazon Web Services (AWS)

The following example creates a serverless index in the `us-west-2`
region of AWS. For more information on serverless and regional availability, see [Understanding indexes](https://docs.pinecone.io/guides/indexes/understanding-indexes#serverless-indexes).

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    Metric,
    VectorType
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_WEST_2
    ),
    vector_type=VectorType.DENSE
)
```

#### Create a serverless index on Google Cloud Platform

The following example creates a serverless index in the `us-central1`
region of GCP. For more information on serverless and regional availability, see [Understanding indexes](https://docs.pinecone.io/guides/indexes/understanding-indexes#serverless-indexes).

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    GcpRegion,
    Metric
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.GCP,
        region=GcpRegion.US_CENTRAL1
    )
)
```

#### Create a serverless index on Azure

The following example creates a serverless index on Azure. For more information on serverless and regional availability, see [Understanding indexes](https://docs.pinecone.io/guides/indexes/understanding-indexes#serverless-indexes).

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AzureRegion,
    Metric
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AZURE,
        region=AzureRegion.EASTUS2
    )
)
```

### Create a pod index

The following example creates an index without a metadata
configuration. By default, Pinecone indexes all metadata.

Many of these fields accept string literals if you know
the values you want to use, but enum objects such as
`PodIndexEnvironment`, `PodType`, `Metric` and more can
help you discover available options.

```python
from pinecone import (
    Pinecone,
    PodSpec,
    Metric,
    PodType,
    PodIndexEnvironment
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
pc.create_index(
    name="example-index",
    dimension=1536,
    metric="cosine",
    deletion_protection='enabled',
    spec=PodSpec(
        environment=PodIndexEnvironment.EU_WEST1_GCP,
        pod_type=PodType.P1_X1
    )
)
```

#### Optional configurations for pod indexes

Pod indexes support many optional configuration fields through
the spec object. For example, the following example creates
an index that only indexes the "color" metadata field for queries
with filtering; with this metadata configuration, queries against the
index cannot filter based on any other metadata field.

This example also demonstrates horizontal scaling to multiple `replicas`.
See [Scale pod-based indexes](https://docs.pinecone.io/guides/indexes/pods/scale-pod-based-indexes) for more information on scaling.

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name="example-index-2",
    dimension=1536,
    spec=PodSpec(
        environment="eu-west1-gcp",
        pod_type='p1.x1',
        metadata_config={
            "indexed": ["color"]
        },
        replicas=2
    )
)
```


### Labeling indexes with index tags

Tags are key-value pairs you can attach to indexes to better understand, organize, and identify your resources. Tags are flexible and can be tailored to your needs, but some common use cases for them might be to label an index with the relevant deployment `environment`, `application`, `team`, or `owner`.

Tags can be set during index creation by passing an optional dictionary with the `tags` keyword argument.

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    GcpRegion,
    Metric
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.GCP,
        region=GcpRegion.US_CENTRAL1
    ),
    tags={
        "environment": "testing",
        "owner": "jsmith",
    }
)
```

#### Modifying tags

To modify the tags of an existing index, use `configure_index()`. The `configure_index` method can be used to change the value of an existing tag, add new tags, or delete tags by setting the value to empty string.

Here's an example showing different ways of modifying tags.

#### Viewing tags with describe_index

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

desc = pc.describe_index(name='my-index')
print(desc.tags)
# {
#   'environment': 'testing',
#   'owner': 'jsmith'
# }
```

#### Modifying an existing tag with configure_index

You can modify an existing tag by passing the key-value pair to `configure_index()`. Other tags on the index will not be changed.

````python
pc.configure_index(
    name='my-index',
    tags={"environment": "production"}
)

desc = pc.describe_index(
    name=index_name,
)
print(desc.tags)
# {
#   'environment': 'production',
#   'owner': 'jsmith'
# }
````

#### Adding a new tag to an existing index

Tags passed to the `configure_index` method are merged with any tags that an index already has.

```python
pc.configure_index(
    name='my-index',
    tags={"purpose": "for testing the new chatbot feature"}
)

desc = pc.describe_index(
    name=index_name,
)
print(desc.tags)
# {
#     'purpose': 'for testing the new chatbot feature',
#     'environment': 'production',
#     'owner': 'jsmith'
# }
```

#### Removing a tag

To remove a tag, pass the value empty string.

```python
pc.configure_index(
    name='my-index',
    tags={
        "purpose": "",
        "environment": "",
        "owner": ""
    }
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

The following example deletes the index named `example-index`. Only indexes which are not protected by deletion protection may be deleted.

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

## Configuring deletion protection

If you would like to enable deletion protection, which prevents an index from being deleted, the `configure_index` method also handles that via an optional `deletion_protection` keyword argument.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

# To enable deletion protection
pc.configure_index("example-index", deletion_protection='enabled')

# Disable deletion protection
pc.configure_index("example-index", deletion_protection='disabled')

# Call describe index to verify the configuration change has been applied
desc = pc.describe_index("example-index")
print(desc.deletion_protection)
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

## List vectors

The `list` and `list_paginated` methods can be used to list vector ids matching a particular id prefix.
With clever assignment of vector ids, this can be used to help model hierarchical relationships between
different vectors such as when there are embeddings for multiple chunks or fragments related to the
same document.

The `list` method returns a generator that handles pagination on your behalf.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='xxx')
index = pc.Index(host='hosturl')

# To iterate over all result pages using a generator function
namespace = 'foo-namespace'
for ids in index.list(prefix='pref', limit=3, namespace=namespace):
    print(ids) # ['pref1', 'pref2', 'pref3']

    # Now you can pass this id array to other methods, such as fetch or delete.
    vectors = index.fetch(ids=ids, namespace=namespace)
```

There is also an option to fetch each page of results yourself with `list_paginated`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='xxx')
index = pc.Index(host='hosturl')

# For manual control over pagination
results = index.list_paginated(
    prefix='pref',
    limit=3,
    namespace='foo',
    pagination_token='eyJza2lwX3Bhc3QiOiI5IiwicHJlZml4IjpudWxsfQ=='
)
print(results.namespace) # 'foo'
print([v.id for v in results.vectors]) # ['pref1', 'pref2', 'pref3']
print(results.pagination.next) # 'eyJza2lwX3Bhc3QiOiI5IiwicHJlZml4IjpudWxsfQ=='
print(results.usage) # { 'read_units': 1 }
```

# Collections

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

# Inference API

The Pinecone SDK now supports creating embeddings via the [Inference API](https://docs.pinecone.io/guides/inference/understanding-inference).

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")
model = "multilingual-e5-large"

# Embed documents
text = [
    "Turkey is a classic meat to eat at American Thanksgiving.",
    "Many people enjoy the beautiful mosques in Turkey.",
]
text_embeddings = pc.inference.embed(
    model=model,
    inputs=text,
    parameters={"input_type": "passage", "truncate": "END"},
)

# Upsert documents into Pinecone index

# Embed a query
query = ["How should I prepare my turkey?"]
query_embeddings = pc.inference.embed(
    model=model,
    inputs=query,
    parameters={"input_type": "query", "truncate": "END"},
)

# Send query to Pinecone index to retrieve similar documents
```


# Contributing

If you'd like to make a contribution, or get setup locally to develop the Pinecone Python SDK, please see our [contributing guide](https://github.com/pinecone-io/pinecone-python-client/blob/main/CONTRIBUTING.md)
