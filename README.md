# Pinecone Python SDK
![License](https://img.shields.io/github/license/pinecone-io/pinecone-python-client?color=orange) [![CI](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/pr.yaml/badge.svg)](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/pr.yaml) [![PyPI version](https://img.shields.io/pypi/v/pinecone.svg)](https://pypi.org/project/pinecone/) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

The official Pinecone Python SDK for building vector search applications with AI/ML.

Pinecone is a vector database that makes it easy to add vector search to production applications. Use Pinecone to store, search, and manage high-dimensional vectors for applications like semantic search, recommendation systems, and RAG (Retrieval-Augmented Generation).

## Features

- **Vector Operations**: Store, query, and manage high-dimensional vectors with metadata filtering
- **Serverless & Pod Indexes**: Choose between serverless (auto-scaling) or pod-based (dedicated) indexes
- **Integrated Inference**: Built-in embedding and reranking models for end-to-end search workflows
- **Async Support**: Full asyncio support with `PineconeAsyncio` for modern Python applications
- **GRPC Support**: Optional GRPC transport for improved performance
- **Type Safety**: Full type hints and type checking support

## Table of Contents

- [Documentation](#documentation)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Bringing your own vectors](#bringing-your-own-vectors-to-pinecone)
  - [Bring your own data using Pinecone integrated inference](#bring-your-own-data-using-pinecone-integrated-inference)
- [Pinecone Assistant](#pinecone-assistant)
- [More Information](#more-information-on-usage)
- [Issues & Bugs](#issues--bugs)
- [Contributing](#contributing)

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

- The Pinecone Python SDK requires Python 3.10 or greater. It has been tested with CPython versions from 3.10 to 3.13.
- Before you can use the Pinecone SDK, you must sign up for an account and find your API key in the Pinecone console dashboard at [https://app.pinecone.io](https://app.pinecone.io).

## Installation

The Pinecone Python SDK is distributed on PyPI using the package name `pinecone`. The base installation includes everything you need to get started with vector operations, but you can install optional extras to unlock additional functionality.

**Base installation includes:**
- Core Pinecone client (`Pinecone`)
- Vector operations (upsert, query, fetch, delete)
- Index management (create, list, describe, delete)
- Metadata filtering
- Pinecone Assistant plugin

**Optional extras:**

- `pinecone[asyncio]` - Adds `aiohttp` dependency and enables `PineconeAsyncio` for async/await support. Use this if you're building applications with FastAPI, aiohttp, or other async frameworks.
- `pinecone[grpc]` - Adds `grpcio` and related libraries for GRPC transport. Provides modest performance improvements for data operations like `upsert` and `query`. See the guide on [tuning performance](https://docs.pinecone.io/docs/performance-tuning).

**Configuration:** The SDK can read your API key from the `PINECONE_API_KEY` environment variable, or you can pass it directly when instantiating the client.

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
uv add pinecone

# Install the latest version, optional dependencies
uv add "pinecone[asyncio,grpc]"
```

#### Installing with [poetry](https://python-poetry.org/)

```shell
# Install the latest version
poetry add pinecone

# Install the latest version, with optional dependencies
poetry add pinecone --extras asyncio --extras grpc
```

## Quickstart

### Bringing your own vectors to Pinecone

This example shows how to create an index, add vectors with embeddings you've generated, and query them. This approach gives you full control over your embedding model and vector generation process.

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    VectorType
)

# 1. Instantiate the Pinecone client
# Option A: Pass API key directly
pc = Pinecone(api_key='YOUR_API_KEY')

# Option B: Use environment variable (PINECONE_API_KEY)
# pc = Pinecone()

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

### Bring your own data using Pinecone integrated inference

This example demonstrates using Pinecone's integrated inference capabilities. You provide raw text data, and Pinecone handles embedding generation and optional reranking automatically. This is ideal when you want to focus on your data and let Pinecone handle the ML complexity.

```python
from pinecone import (
    Pinecone,
    CloudProvider,
    AwsRegion,
    EmbedModel,
    IndexEmbed,
)

# 1. Instantiate the Pinecone client
# The API key can be passed directly or read from PINECONE_API_KEY environment variable
pc = Pinecone(api_key='YOUR_API_KEY')

# 2. Create an index configured for use with a particular embedding model
# This sets up the index with the right dimensions and configuration for your chosen model
index_config = pc.create_index_for_model(
    name="my-model-index",
    cloud=CloudProvider.AWS,
    region=AwsRegion.US_EAST_1,
    embed=IndexEmbed(
        model=EmbedModel.Multilingual_E5_Large,
        field_map={"text": "my_text_field"}
    )
)

# 3. Instantiate an Index client for data operations
idx = pc.Index(host=index_config.host)

# 4. Upsert records with raw text data
# Pinecone will automatically generate embeddings using the configured model
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

# 5. Search for similar records using text queries
# Pinecone handles embedding the query and optionally reranking results
from pinecone import SearchQuery, SearchRerank, RerankModel

response = idx.search_records(
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

## Pinecone Assistant
### Installing the Pinecone Assistant Python plugin

The `pinecone-plugin-assistant` package is now bundled by default when installing `pinecone`. It does not need to be installed separately in order to use Pinecone Assistant.

For more information on Pinecone Assistant, see the [Pinecone Assistant documentation](https://docs.pinecone.io/guides/assistant/overview).


## More information on usage

Detailed information on specific ways of using the SDK are covered in these guides:

**Index Management:**
- [Serverless Indexes](./docs/db_control/serverless-indexes.md) - Learn about auto-scaling serverless indexes that scale automatically with your workload
- [Pod Indexes](./docs/db_control/pod-indexes.md) - Understand dedicated pod-based indexes for consistent performance

**Data Operations:**
- [Working with vectors](./docs/db_data/index-usage-byov.md) - Comprehensive guide to storing, querying, and managing vectors with metadata filtering

**Advanced Features:**
- [Inference API](./docs/inference-api.md) - Use Pinecone's integrated embedding and reranking models
- [FAQ](./docs/faq.md) - Common questions and troubleshooting tips


# Issues & Bugs

If you notice bugs or have feedback, please [file an issue](https://github.com/pinecone-io/pinecone-python-client/issues).

You can also get help in the [Pinecone Community Forum](https://community.pinecone.io/).

# Contributing

If you'd like to make a contribution, or get setup locally to develop the Pinecone Python SDK, please see our [contributing guide](https://github.com/pinecone-io/pinecone-python-client/blob/main/CONTRIBUTING.md)
