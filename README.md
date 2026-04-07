# Pinecone Python SDK

The Pinecone Python SDK provides a client for the [Pinecone](https://www.pinecone.io/) vector database. Use it to create and manage indexes, upsert and query vectors, and run inference operations from Python.

Requires Python 3.10+.

## Installation

```bash
pip install pinecone
```

For development dependencies (testing, type checking, linting):

```bash
pip install pinecone[dev]
```

## Quick start

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize the client
pc = Pinecone(api_key="your-api-key")

# Create a serverless index
pc.create_index(
    name="movie-recommendations",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Connect to the index
index = pc.Index("movie-recommendations")

# Upsert vectors
index.upsert(
    vectors=[
        ("movie-42", [0.012, -0.087, 0.153, ...]),  # 1536-dim embedding
        ("movie-87", [0.045, 0.021, -0.064, ...]),
    ],
    namespace="movies-en",
)

# Query for similar vectors
results = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=10,
    namespace="movies-en",
)

for match in results.matches:
    print(f"{match.id}: {match.score:.4f}")
```

## Async usage

The SDK provides an async client for use with `asyncio`:

```python
import asyncio
from pinecone import AsyncPinecone

async def main():
    pc = AsyncPinecone(api_key="your-api-key")
    index = pc.Index("movie-recommendations")

    results = await index.query(
        vector=[0.012, -0.087, 0.153, ...],
        top_k=10,
        namespace="movies-en",
    )
    for match in results.matches:
        print(f"{match.id}: {match.score:.4f}")

asyncio.run(main())
```

## Configuration

### API key

Pass the API key directly or set the `PINECONE_API_KEY` environment variable:

```python
from pinecone import Pinecone

# Explicit API key
pc = Pinecone(api_key="your-api-key")

# From environment variable (PINECONE_API_KEY)
pc = Pinecone()
```

### Custom host

Connect to a specific control plane host:

```python
pc = Pinecone(api_key="your-api-key", host="https://api.pinecone.io")
```

### Timeout

Configure request timeouts in seconds:

```python
pc = Pinecone(api_key="your-api-key", timeout=30)
```

### Debug logging

Enable debug logging by setting the `PINECONE_DEBUG` environment variable:

```bash
export PINECONE_DEBUG=1
```

## Development

### Setup

Clone the repository and install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### Tests

```bash
uv run pytest tests/unit/ -x -v
```

### Type checking

```bash
uv run mypy --strict pinecone/
```

### Linting and formatting

```bash
uv run ruff check --fix
uv run ruff format
```

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
