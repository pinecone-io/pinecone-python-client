# Pinecone Python SDK 
![License](https://img.shields.io/github/license/pinecone-io/pinecone-python-client?color=orange) [![CI](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/pr.yaml/badge.svg)](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/pr.yaml)

The official Pinecone Python SDK.

For more information, see the docs at https://docs.pinecone.io

## Documentation

- [**Reference Documentation**](https://sdk.pinecone.io/python/index.html)

### Upgrading the SDK

#### Upgrading from `4.x` to `5.x` 

As part of an overall move to stop exposing generated code in the package's public interface, an obscure configuration property (`openapi_config`) was removed in favor of individual configuration options such as `proxy_url`, `proxy_headers`, and `ssl_ca_certs`. All of these properties were available in v3 and v4 releases of the SDK, with deprecation notices shown to affected users.

It is no longer necessary to install a separate plugin, `pinecone-plugin-inference`, to try out the [Inference API](https://docs.pinecone.io/guides/inference/understanding-inference); that plugin is now installed by default in the v5 SDK. See [usage instructions below](#inference-api).

#### Older releases

- **Upgrading to `4.x`** : For this upgrade you are unlikely to be impacted by breaking changes unless you are using the `grpc` extras (see install steps below). Read full details in these [v4 Release Notes](https://github.com/pinecone-io/pinecone-python-client/releases/tag/v4.0.0).

- **Upgrading to `3.x`**: Many things were changed in the v3 SDK to pave the way for Pinecone's new Serverless index offering. These changes are covered in detail in the [**v3 Migration Guide**](https://canyon-quilt-082.notion.site/Pinecone-Python-SDK-v3-0-0-Migration-Guide-056d3897d7634bf7be399676a4757c7b#a21aff70b403416ba352fd30e300bce3). Serverless indexes are only available in `3.x` release versions or greater.

### Example code

Many of the brief examples shown in this README are using very small vectors to keep the documentation concise, but most real world usage will involve much larger embedding vectors. To see some more realistic examples of how this SDK can be used, explore some of our many Jupyter notebooks in the [examples](https://github.com/pinecone-io/examples) repository.

## Prerequisites

The Pinecone Python SDK is compatible with Python 3.8 and greater.

## Installation

There are two flavors of the Pinecone Python SDK. The default flavor installed from PyPI as `pinecone` has a minimal set of dependencies and interacts with Pinecone via HTTP requests.

If you are aiming to maximimize performance, you can install additional gRPC dependencies to access an alternate SDK implementation that relies on gRPC for data operations. See the guide on [tuning performance](https://docs.pinecone.io/docs/performance-tuning).

### Installing with pip

```shell
# Install the latest version
pip3 install pinecone

# Install the latest version, with extra grpc dependencies
pip3 install "pinecone[grpc]"

# Install a specific version
pip3 install pinecone==5.0.0

# Install a specific version, with grpc extras
pip3 install "pinecone[grpc]"==5.0.0
```

### Installing with poetry

```shell
# Install the latest version
poetry add pinecone

# Install the latest version, with grpc extras
poetry add pinecone --extras grpc

# Install a specific version
poetry add pinecone==5.0.0

# Install a specific version, with grpc extras
poetry add pinecone==5.0.0 --extras grpc
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

### Proxy configuration

If your network setup requires you to interact with Pinecone via a proxy, you will need
to pass additional configuration using optional keyword parameters. These optional parameters are forwarded to `urllib3`, which is the underlying library currently used by the Pinecone SDK to make HTTP requests. You may find it helpful to refer to the [urllib3 documentation on working with proxies](https://urllib3.readthedocs.io/en/stable/advanced-usage.html#http-and-https-proxies) while troubleshooting these settings.

Here is a basic example:

```python
from pinecone import Pinecone

pc = Pinecone(
    api_key='YOUR_API_KEY',
    proxy_url='https://your-proxy.com'
)

pc.list_indexes()
```

If your proxy requires authentication, you can pass those values in a header dictionary using the `proxy_headers` parameter.

```python
from pinecone import Pinecone
import urllib3 import make_headers

pc = Pinecone(
    api_key='YOUR_API_KEY',
    proxy_url='https://your-proxy.com',
    proxy_headers=make_headers(proxy_basic_auth='username:password')
)

pc.list_indexes()
```

### Using proxies with self-signed certificates

By default the Pinecone Python SDK will perform SSL certificate verification
using the CA bundle maintained by Mozilla in the [certifi](https://pypi.org/project/certifi/) package.

If your proxy server is using a self-signed certificate, you will need to pass the path to the certificate in PEM format using the `ssl_ca_certs` parameter.

```python
from pinecone import Pinecone
import urllib3 import make_headers

pc = Pinecone(
    api_key="YOUR_API_KEY",
    proxy_url='https://your-proxy.com',
    proxy_headers=make_headers(proxy_basic_auth='username:password'),
    ssl_ca_certs='path/to/cert-bundle.pem'
)

pc.list_indexes()
```

### Disabling SSL verification

If you would like to disable SSL verification, you can pass the `ssl_verify`
parameter with a value of `False`. We do not recommend going to production with SSL verification disabled.

```python
from pinecone import Pinecone
import urllib3 import make_headers

pc = Pinecone(
    api_key='YOUR_API_KEY',
    proxy_url='https://your-proxy.com',
    proxy_headers=make_headers(proxy_basic_auth='username:password'),
    ssl_ca_certs='path/to/cert-bundle.pem',
    ssl_verify=False
)

pc.list_indexes()

```

### Working with GRPC (for improved performance)

If you've followed instructions above to install with optional `grpc` extras, you can unlock some performance improvements by working with an alternative version of the SDK imported from the `pinecone.grpc` subpackage.

```python
import os
from pinecone.grpc import PineconeGRPC

pc = PineconeGRPC(api_key=os.environ.get('PINECONE_API_KEY'))

# From here on, everything is identical to the REST-based SDK.
index = pc.Index(host='my-index-8833ca1.svc.us-east1-gcp.pinecone.io')

index.upsert(vectors=[])
index.query(vector=[...], top_key=10)
```

# Indexes

## Create Index

### Create a serverless index

The following example creates a serverless index in the `us-west-2`
region of AWS. For more information on serverless and regional availability, see [Understanding indexes](https://docs.pinecone.io/guides/indexes/understanding-indexes#serverless-indexes).

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
pc.create_index(
    name='my-index',
    dimension=1536,
    metric='euclidean',
    deletion_protection='enabled',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-west-2'
    )
)
```

### Create a pod index

The following example creates an index without a metadata
configuration. By default, Pinecone indexes all metadata.

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
pc.create_index(
    name="example-index",
    dimension=1536,
    metric="cosine",
    deletion_protection='enabled',
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
