# Pinecone Python SDK
![License](https://img.shields.io/github/license/pinecone-io/pinecone-python-client?color=orange) [![CI](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/pr.yaml/badge.svg)](https://github.com/pinecone-io/pinecone-python-client/actions/workflows/pr.yaml)

The official Pinecone Python SDK.

For more information, see the docs at https://docs.pinecone.io


## Documentation

- [**Reference Documentation**](https://sdk.pinecone.io/python/index.html)

### Upgrading the SDK

For notes on changes between major versions, see [Upgrading](./docs/upgrading.md)

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

Bring your own vector:
- [Serverless Indexes](./docs/serverless-indexes.md)
- [Pod Indexes](./docs/pod-indexes.md)


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
