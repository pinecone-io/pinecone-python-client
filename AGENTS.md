# Pinecone Python SDK

The Pinecone Python SDK provides access to the Pinecone vector database. Use `Pinecone` for control-plane operations (creating and managing indexes) and `Index` for data-plane operations (upserting and querying vectors). The `pc.index("name")` method bridges the two.

## Quick Start

```python
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create a serverless index
pc.indexes.create(
    name="movie-recommendations",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Get a handle to the index (data-plane operations)
index = pc.index("movie-recommendations")

# Upsert vectors
index.upsert(vectors=[
    ("movie-42", [0.012, -0.087, 0.153, ...]),
    ("movie-43", [0.045, 0.021, -0.064, ...]),
])

# Query by vector similarity
results = index.query(vector=[0.012, -0.087, 0.153, ...], top_k=5)
for match in results.matches:
    print(match.id, match.score)
```

## Key Classes

| Class | Import | Purpose |
|---|---|---|
| `Pinecone` | `from pinecone import Pinecone` | Sync client for control-plane operations (indexes, collections, backups) |
| `AsyncPinecone` | `from pinecone import AsyncPinecone` | Async variant of `Pinecone` for use with `asyncio` |
| `Index` | Obtained via `pc.index("name")` | Sync client for data-plane operations (upsert, query, delete, fetch) |
| `AsyncIndex` | Obtained via `async_pc.index("name")` | Async variant of `Index` |
| `Admin` | `from pinecone import Admin` | Organization and project management via OAuth2 credentials |

## Control Plane vs Data Plane

`Pinecone` manages indexes, collections, and backups. It talks to the Pinecone control-plane API. `Index` performs vector operations (upsert, query, fetch, delete) against a specific index. It talks to the data-plane API hosted on the index's own endpoint. Call `pc.index("name")` to get an `Index` handle from a `Pinecone` client. The two use different hosts and authentication scopes.

## Common Workflows

### Store and retrieve vectors

```python
from pinecone import Pinecone, Vector

pc = Pinecone(api_key="your-api-key")
index = pc.index("article-search")

index.upsert(vectors=[
    Vector(id="article-101", values=[0.012, -0.087, 0.153, ...],
           metadata={"topic": "science", "year": 2024}),
])

results = index.query(
    vector=[0.012, -0.087, 0.153, ...], top_k=10,
    filter={"topic": "science"}, namespace="articles-en",
)
```

### Semantic search with integrated embeddings

```python
from pinecone import Pinecone, IntegratedSpec, EmbedConfig, EmbedModel

pc = Pinecone(api_key="your-api-key")
pc.indexes.create(
    name="product-catalog",
    spec=IntegratedSpec(cloud="aws", region="us-east-1",
        embed=EmbedConfig(model=EmbedModel.MULTILINGUAL_E5_LARGE,
                          field_map={"text": "description"})),
)
index = pc.index("product-catalog")
index.upsert_records(namespace="products", records=[
    {"id": "prod-001", "description": "Lightweight running shoes", "category": "footwear"},
])
results = index.search_records(namespace="products",
    query={"inputs": {"text": "comfortable shoes for trail running"}, "top_k": 5})
```

### Generate embeddings and rerank

```python
pc = Pinecone(api_key="your-api-key")

# Embed text
response = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["How do I reset my password?"],
    parameters={"input_type": "query"},
)

# Rerank documents by relevance
result = pc.inference.rerank(
    model="pinecone-rerank-v0", query="best budget laptop", top_n=2,
    documents=["Affordable laptops under $500", "Premium gaming desktops"],
)
```

## Error Handling

All SDK exceptions inherit from `PineconeError`:

```
PineconeError
├── ApiError              # HTTP error from the API (has .status, .body)
│   ├── NotFoundError     # 404
│   ├── UnauthorizedError # 401
│   ├── ForbiddenError    # 403
│   ├── ConflictError     # 409
│   └── ServiceError      # 5xx
├── PineconeValueError    # Invalid argument (also a ValueError)
├── PineconeTypeError     # Wrong type (also a TypeError)
├── PineconeTimeoutError  # Request timed out
├── PineconeConnectionError # Network connectivity failure
└── ResponseParsingError  # Unexpected response format
```

Catch specific exceptions (`NotFoundError`, `UnauthorizedError`, etc.) or the base `ApiError` for HTTP errors. `ApiError` exposes `.status` and `.body` attributes.

## Common Mistakes

- **Calling `pc.upsert()` instead of `index.upsert()`** — upsert, query, fetch, and delete are on the `Index` object, not on `Pinecone`. Use `index = pc.index("name")` first.
- **Not waiting for index readiness** — a freshly created index is not immediately ready. Use `pc.indexes.describe("name")` and check `status.ready` before upserting, or pass `wait_until_ready=True` when creating.
- **Forgetting the namespace** — vectors in different namespaces are isolated. If you upsert to `namespace="articles-en"` but query without specifying a namespace, you query the default (`""`) namespace and get no results.
- **Mismatched vector dimensions** — the vector length in upsert and query must match the index's `dimension`. The API returns an error if they differ.
- **Using `from pinecone import Index` directly** — `Index` requires a host URL. Use `pc.index("name")` to resolve the host automatically.
