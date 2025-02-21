# Pinecone Python SDK
![License](https://img.shields.io/github/license/pinecone-io/pinecone-python-client?color=orange) [![CI](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/pr.yaml/badge.svg)](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/pr.yaml)

The official Pinecone Python SDK.

## Documentation

- [**Conceptual docs and guides**](https://docs.pinecone.io)
- [**Python Reference Documentation**](https://sdk.pinecone.io/python/index.html)

### Upgrading the SDK

> [!NOTE]
> The official SDK package was renamed from `pinecone-client` to `pinecone` beginning in version `5.1.0`.
> Please remove `pinecone-client` from your project dependencies and add `pinecone` instead to get
> the latest updates.

For notes on changes between major versions, see [Upgrading](./docs/upgrading.md)

## Prerequisites

- The Pinecone Python SDK is compatible with Python 3.9 and greater. It has been tested with CPython versions from 3.9 to 3.13.
- Before you can use the Pinecone SDK, you must sign up for an account and find your API key in the Pinecone console dashboard at [https://app.pinecone.io](https://app.pinecone.io).

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

# Quickstart

## Bringing your own vectors to Pinecone

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    VectorType
)

# 1. Instantiate the Pinecone client
pc = Pinecone(api_key='YOUR_API_KEY')

# 2. Create an index
index_config = pc.create_index(
    name="index-name",
    dimension=1536,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1
    ),
    vector_type=VectorType.DENSE
)

# 3. Instantiate an Index client
idx = pc.Index(host=index_config.host)

# 4. Upsert embeddings
idx.upsert(
    vectors=[
        ("id1", [0.1, 0.2, 0.3, 0.4, ...], {"metadata_key": "value1"}),
        ("id2", [0.2, 0.3, 0.4, 0.5, ...], {"metadata_key": "value2"}),
    ],
    namespace="example-namespace"
)

# 5. Query your index using an embedding
query_embedding = [...] # list should have length == index dimension
idx.query(
    vector=query_embedding,
    top_k=10,
    include_metadata=True,
    filter={"metadata_key": { "$eq": "value1" }}
)
```

## Bring your own data using Pinecone integrated inference

```python
from pinecone import (
    Pinecone,
    CloudProvider,
    AwsRegion,
    EmbedModel,
)

# 1. Instantiate the Pinecone client
pc = Pinecone(api_key="<<PINECONE_API_KEY>>")

# 2. Create an index configured for use with a particular model
index_config = pc.create_index_for_model(
    name="my-model-index",
    cloud=CloudProvider.AWS,
    region=AwsRegion.US_EAST_1,
    embed=IndexEmbed(
        model=EmbedModel.Multilingual_E5_Large,
        field_map={"text": "my_text_field"}
    )
)

# 3. Instantiate an Index client
idx = pc.Index(host=index_config.host)

# 4. Upsert records
idx.upsert_records(
    namespace="my-namespace",
    records=[
        {
            "_id": "test1",
            "my_text_field": "Apple is a popular fruit known for its sweetness and crisp texture.",
        },
        {
            "_id": "test2",
            "my_text_field": "The tech company Apple is known for its innovative products like the iPhone.",
        },
        {
            "_id": "test3",
            "my_text_field": "Many people enjoy eating apples as a healthy snack.",
        },
        {
            "_id": "test4",
            "my_text_field": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
        },
        {
            "_id": "test5",
            "my_text_field": "An apple a day keeps the doctor away, as the saying goes.",
        },
        {
            "_id": "test6",
            "my_text_field": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership.",
        },
    ],
)

# 5. Search for similar records
from pinecone import SearchQuery, SearchRerank, RerankModel

response = index.search_records(
    namespace="my-namespace",
    query=SearchQuery(
        inputs={
            "text": "Apple corporation",
        },
        top_k=3
    ),
    rerank=SearchRerank(
        model=RerankModel.Bge_Reranker_V2_M3,
        rank_fields=["my_text_field"],
        top_n=3,
    ),
)
```

## More information on usage

Detailed information on specific ways of using the SDK are covered in these other pages.

- Store and query your vectors
  - [Serverless Indexes](./docs/db_control/serverless-indexes.md)
  - [Pod Indexes](./docs/db_control/pod-indexes.md)
  - [Working with vectors](./docs/db_data/index-usage-byov.md)

- [Inference API](./docs/inference-api.md)
- [FAQ](./docs/faq.md)


# Issues & Bugs

If you notice bugs or have feedback, please [file an issue](https://github.com/pinecone-io/pinecone-python-client/issues).

You can also get help in the [Pinecone Community Forum](https://community.pinecone.io/).

# Contributing

If you'd like to make a contribution, or get setup locally to develop the Pinecone Python SDK, please see our [contributing guide](https://github.com/pinecone-io/pinecone-python-client/blob/main/CONTRIBUTING.md)
