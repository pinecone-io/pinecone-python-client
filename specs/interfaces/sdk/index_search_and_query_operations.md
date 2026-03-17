# Index Vector Search and Query Operations

This module documents the Index class methods for searching and querying vectors in Pinecone indexes: `search()` for embedding-based search with integrated inference, and `query()` for direct vector similarity search with dense or sparse vectors, optional reranking, metadata filtering, and async execution support.

Vector search and query operations are accessed through an Index client, obtained from the Pinecone client:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index(host="your-index-host")

# Synchronous search/query
response = index.search(namespace="my-namespace", query={"text": "query terms"}, ...)
response = index.query(vector=[...], top_k=10, namespace="my-namespace", ...)

# Asynchronous search/query
async_index = pc.IndexAsyncio(host="your-index-host")
response = await async_index.search(namespace="my-namespace", query={"text": "query terms"}, ...)
response = await async_index.query(vector=[...], top_k=10, namespace="my-namespace", ...)
```

---

## `Index.search()`

Searches for records in a namespace using an embedding model. This operation converts a query to a vector embedding and then searches the namespace. Requires an index with integrated inference configured.

**Source:** `pinecone/db_data/index.py:660-744`, `pinecone/db_data/index_asyncio.py:1400-1501` (async equivalent)

**Added:** v8.1
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** Queries the index; no mutations

### Signature

```python
def search(
    self,
    namespace: str,
    query: SearchQueryTypedDict | SearchQuery,
    rerank: SearchRerankTypedDict | SearchRerank | None = None,
    fields: list[str] | None = ["*"]
) -> SearchRecordsResponse
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `namespace` | `str` | Yes | — | v8.1 | No | The namespace in the index to search. |
| `query` | `SearchQueryTypedDict \| SearchQuery` | Yes | — | v8.1 | No | The search query object containing `inputs` (dict) and `top_k` (int). May optionally include `filter`, `vector`, `id`, or `match_terms`. See **SearchQuery Details** below. |
| `rerank` | `SearchRerankTypedDict \| SearchRerank \| None` | No | `None` | v8.1 | No | Optional reranking configuration to apply to search results. Specifies the reranking model, fields to rank against, and optionally the number of top results to return. See **SearchRerank Details** below. |
| `fields` | `list[str] \| None` | No | `["*"]` | v8.1 | No | List of fields to return in the response. Use `["*"]` to return all fields. |

### SearchQuery Details

`SearchQuery` is a dataclass representing the query when searching within a specific namespace. It can be constructed as a dictionary (TypedDict) or as a `SearchQuery` object.

**SearchQuery fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| inputs | dict[str, Any] | Yes | The input data to search with. Typically `{"text": "search query"}` or `{"text": "<field_name>", "<field_name>": "value", ...}` depending on the model's input requirements. |
| top_k | int | Yes | The number of results to return with each search. Must be a positive integer. |
| filter | dict[str, Any] \| None | No | Optional metadata filter to apply to the search. See metadata filtering documentation. |
| vector | SearchQueryVectorTypedDict \| SearchQueryVector \| None | No | Optional vector values to search with. If provided, overwrites the `inputs`. |
| id | str \| None | No | Optional unique ID of a vector to use as a query vector. If provided, overwrites the `inputs`. |
| match_terms | dict[str, Any] \| None | No | Optional matching strategy and terms. Format: `{"strategy": "all", "terms": ["term1", "term2"]}`. Currently only "all" strategy is supported (all terms must be present). **Limitation:** Only supported for sparse indexes with integrated embedding configured to use the `pinecone-sparse-english-v0` model. |

### SearchRerank Details

`SearchRerank` is a dataclass representing reranking configuration. It can be constructed as a dictionary (TypedDict) or as a `SearchRerank` object.

**SearchRerank fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model | str \| RerankModel | Yes | The name of the reranking model to use. Use string value or `RerankModel` enum (e.g., `RerankModel.Bge_Reranker_V2_M3`). |
| rank_fields | list[str] | Yes | The list of fields to use for reranking. These fields must exist in the record metadata. |
| top_n | int \| None | No | The number of top results to return after reranking. Defaults to `top_k`. |
| parameters | dict[str, Any] \| None | No | Optional additional model-specific parameters. Refer to the reranking model documentation for available parameters. |
| query | str \| None | No | Optional custom query to rerank documents against. If specified, overwrites the query input provided at the top level. |

### Returns

**Type:** `SearchRecordsResponse` — A response object containing:
- `results` (list[dict]) — List of matched records with their data and metadata.
- `usage` (dict) — Usage statistics from the search operation.
- Additional metadata about the search execution.

### Raises

| Exception | Condition |
|-----------|-----------|
| `Exception` | The `namespace` parameter is `None` or missing. |
| `ValueError` | The query or rerank parameters are invalid. |
| `ProtocolError` | Connection failed to the index host. |
| `BadRequestException` | Invalid request parameters or model configuration. |
| `NotFoundException` | The namespace does not exist. |
| `PineconeApiException` | Unexpected server error. |

### Example

```python
from pinecone import (
    Pinecone,
    CloudProvider,
    AwsRegion,
    EmbedModel,
    IndexEmbed,
    SearchQuery,
    SearchRerank,
    RerankModel
)

# Create index with integrated embedding
pc = Pinecone(api_key="your-api-key")
index_model = pc.create_index_for_model(
    name="my-model-index",
    cloud=CloudProvider.AWS,
    region=AwsRegion.US_WEST_2,
    embed=IndexEmbed(
        model=EmbedModel.Multilingual_E5_Large,
        field_map={"text": "my_text_field"}
    )
)

# Get index client
idx = pc.Index(host=index_model.host)

# Search with reranking
response = idx.search(
    namespace="my-namespace",
    query=SearchQuery(
        inputs={"text": "Apple corporation"},
        top_k=10,
    ),
    rerank=SearchRerank(
        model=RerankModel.Bge_Reranker_V2_M3,
        rank_fields=["my_text_field"],
        top_n=3,
    ),
)

for result in response.results:
    print(f"ID: {result['id']}, Score: {result['score']}")

# Search using dictionary syntax (alternative)
response = idx.search(
    namespace="my-namespace",
    query={
        "inputs": {"text": "Apple corporation"},
        "top_k": 10,
    },
    rerank={
        "model": "bge-reranker-v2-m3",
        "rank_fields": ["my_text_field"],
        "top_n": 3,
    }
)
```

### Notes

- The `search()` method requires an index with integrated inference enabled. The embedding model is configured when the index is created.
- The `field_map` configuration maps model input fields to actual record fields.
- `match_terms` is a specialized feature for keyword matching and is only available for sparse indexes with the `pinecone-sparse-english-v0` embedding model.
- If no `fields` are specified, the default is `["*"]`, which returns all fields.
- Reranking is optional; if omitted, results are returned in their original similarity order.

---

## `Index.search_records()`

Alias of `search()`. Provides the same functionality with an alternative method name.

**Source:** `pinecone/db_data/index.py:746-759`, `pinecone/db_data/index_asyncio.py:1503-1515` (async equivalent)

**Added:** v8.1
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** Queries the index; no mutations

### Signature

```python
def search_records(
    self,
    namespace: str,
    query: SearchQueryTypedDict | SearchQuery,
    rerank: SearchRerankTypedDict | SearchRerank | None = None,
    fields: list[str] | None = ["*"]
) -> SearchRecordsResponse
```

See `Index.search()` for complete documentation of parameters, returns, raises, and examples.

---

## `Index.query()`

Queries a namespace using a query vector. This operation retrieves the IDs and similarity scores of the most similar items in a namespace. Supports dense vectors, sparse vectors (for hybrid search), metadata filtering, and async execution.

**Source:** `pinecone/db_data/index.py:936-1056`, `pinecone/db_data/index_asyncio.py:720-905` (async equivalent)

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** Queries the index; no mutations

### Signature

```python
def query(
    self,
    top_k: int,
    vector: list[float] | None = None,
    id: str | None = None,
    namespace: str | None = None,
    filter: FilterTypedDict | None = None,
    include_values: bool | None = None,
    include_metadata: bool | None = None,
    sparse_vector: SparseValues | SparseVectorTypedDict | None = None,
    scan_factor: float | None = None,
    max_candidates: int | None = None,
    **kwargs
) -> QueryResponse | ApplyResult
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `top_k` | `int` | Yes | — | v8.0 | No | The number of results to return for each query. Must be an integer greater than or equal to 1. |
| `vector` | `list[float] \| None` | No | `None` | v8.0 | No | The query vector (dense embedding). Must be the same length as the index dimension. Cannot be used together with `id`. |
| `id` | `str \| None` | No | `None` | v8.0 | No | The unique ID of an existing vector to use as the query vector. Cannot be used together with `vector`. |
| `namespace` | `str \| None` | No | `None` | v8.0 | No | The namespace to query. If omitted, the default (empty string) namespace is used. |
| `filter` | `FilterTypedDict \| None` | No | `None` | v8.0 | No | Optional metadata filter to apply. Limits results to vectors whose metadata matches the filter. See **Filter Details** below. |
| `include_values` | `bool \| None` | No | `None` | v8.0 | No | Whether vector values should be included in the response. Default behavior (None) means values are not included. |
| `include_metadata` | `bool \| None` | No | `None` | v8.0 | No | Whether metadata should be included in the response. Default behavior (None) means metadata is not included. |
| `sparse_vector` | `SparseValues \| SparseVectorTypedDict \| None` | No | `None` | v8.0 | No | Optional sparse vector for hybrid search. Expected format: `{"indices": list[int], "values": list[float]}` or a `SparseValues` object. |
| `scan_factor` | `float \| None` | No | `None` | v8.0 | No | Optimization parameter for IVF dense indexes in dedicated read node (DRN) indexes. Adjusts how much of the index is scanned (range: 0.5-4, default 1.0). Only supported for DRN dense indexes. |
| `max_candidates` | `int \| None` | No | `None` | v8.0 | No | Optimization parameter for DRN dense indexes. Controls the maximum number of candidate dense vectors to rerank (range: top_k to 100,000). Only supported for DRN dense indexes. |
| `**kwargs` | `dict` | No | — | v8.0 | No | Additional keyword arguments for the API call. Supported kwargs: `async_req` (bool) — if True, returns `ApplyResult` instead of `QueryResponse`; `async_threadpool_executor` (bool) — if True, uses thread pool for async execution. |

### Filter Details

Filters use a MongoDB-like query syntax with support for comparison operators and logical operators.

**Simple filter examples:**

```python
# Exact match
{"genre": "drama"}

# Comparison operators
{"year": {"$gt": 2000}}        # Greater than
{"rating": {"$gte": 7.5}}      # Greater than or equal
{"views": {"$lt": 1000000}}    # Less than
{"votes": {"$lte": 100}}       # Less than or equal
{"status": {"$eq": "active"}}  # Equal
{"status": {"$ne": "deleted"}} # Not equal

# Multiple values
{"year": {"$in": [2020, 2021, 2022]}}
{"status": {"$nin": ["draft", "archived"]}}

# Existence
{"optional_field": {"$exists": True}}
```

**Logical operators (combine multiple conditions):**

```python
# AND - all conditions must be true
{"$and": [{"genre": "drama"}, {"year": {"$gt": 2000}}]}

# OR - at least one condition must be true
{"$or": [{"genre": "drama"}, {"genre": "comedy"}]}
```

### Returns

**Type:** `QueryResponse` (default) — A response object containing:
- `matches` (list[ScoredVector]) — List of matched vectors with IDs and similarity scores.
  - `id` (str) — The vector ID.
  - `score` (float) — The similarity score.
  - `values` (list[float]) — The vector values (only if `include_values=True`).
  - `metadata` (dict) — The vector metadata (only if `include_metadata=True`).
- `namespace` (str) — The namespace that was queried.
- `usage` (dict \| None) — Token usage statistics.

When `async_req=True` or `async_threadpool_executor=True`, returns `ApplyResult` — An async result wrapper. Call `.get()` to retrieve the actual `QueryResponse` (may block).

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `top_k` is less than 1, or both `vector` and `id` are provided, or neither `vector` nor `id` are provided. |
| `ProtocolError` | Connection failed to the index host. |
| `BadRequestException` | Invalid filter syntax or malformed request. |
| `NotFoundException` | The namespace does not exist or the vector ID does not exist. |
| `PineconeApiException` | Unexpected server error. |

### Example

```python
from pinecone import Pinecone, SparseValues

pc = Pinecone(api_key="your-api-key")
index = pc.Index(host="your-index-host")

# Query with a dense vector
response = index.query(
    vector=[0.1, 0.2, 0.3, 0.4, 0.5],
    top_k=10,
    namespace="my-namespace"
)

for match in response.matches:
    print(f"ID: {match.id}, Score: {match.score}")

# Query using an existing vector ID
response = index.query(
    id="vector-id-1",
    top_k=10,
    namespace="my-namespace"
)

# Query with metadata filtering
response = index.query(
    vector=[0.1, 0.2, 0.3, 0.4, 0.5],
    top_k=10,
    namespace="my-namespace",
    filter={"genre": {"$eq": "drama"}, "year": {"$gt": 2000}}
)

# Query with include_values and include_metadata
response = index.query(
    id="vector-id-1",
    top_k=10,
    namespace="my-namespace",
    include_metadata=True,
    include_values=True
)

for match in response.matches:
    print(f"ID: {match.id}, Score: {match.score}")
    print(f"Metadata: {match.metadata}")
    print(f"Values: {match.values}")

# Hybrid search (dense + sparse vectors)
response = index.query(
    vector=[0.1, 0.2, 0.3, 0.4, 0.5],
    sparse_vector={"indices": [10, 20, 30], "values": [0.1, 0.2, 0.3]},
    top_k=10,
    namespace="my-namespace"
)

# Hybrid search using SparseValues object
response = index.query(
    vector=[0.1, 0.2, 0.3, 0.4, 0.5],
    sparse_vector=SparseValues(indices=[10, 20, 30], values=[0.1, 0.2, 0.3]),
    top_k=10,
    namespace="my-namespace"
)

# Asynchronous query (with ApplyResult)
import time
result = index.query(
    vector=[0.1, 0.2, 0.3, 0.4, 0.5],
    top_k=10,
    namespace="my-namespace",
    async_req=True
)

# Block until result is ready
response = result.get(timeout=30)
```

### Notes

- Exactly one of `vector` or `id` must be provided. Providing both or neither raises `ValueError`.
- `top_k` must be greater than or equal to 1.
- Metadata filters use MongoDB-like syntax with operators: `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`, `$in`, `$nin`, `$exists`, `$and`, `$or`.
- `scan_factor` and `max_candidates` are only effective for DRN (dedicated read node) dense indexes and are ignored for serverless or pod-based indexes.
- For hybrid search, both `vector` and `sparse_vector` should be provided. The index must support both dense and sparse vectors.
- When using `async_req=True`, the method returns immediately with an `ApplyResult`. Call `.get()` on it to retrieve the actual response (potentially blocking).
- The `scan_factor` and `max_candidates` parameters optimize query performance by controlling how many candidates are considered during the search phase. Higher values increase recall but add query latency.

---

## Async Variants

### `AsyncIndex.search()`

Async variant of `Index.search()`. Returns a coroutine that must be awaited.

**Source:** `pinecone/db_data/index_asyncio.py:1400-1501`

**Added:** v8.1
**Deprecated:** No

#### Signature

```python
async def search(
    self,
    namespace: str,
    query: SearchQueryTypedDict | SearchQuery,
    rerank: SearchRerankTypedDict | SearchRerank | None = None,
    fields: list[str] | None = ["*"]
) -> SearchRecordsResponse
```

#### Example

```python
import asyncio
from pinecone import Pinecone, SearchQuery, SearchRerank, RerankModel

async def main():
    pc = Pinecone(api_key="your-api-key")
    async_index = pc.IndexAsyncio(host="your-index-host")

    response = await async_index.search(
        namespace="my-namespace",
        query=SearchQuery(
            inputs={"text": "Apple corporation"},
            top_k=10,
        ),
        rerank=SearchRerank(
            model=RerankModel.Bge_Reranker_V2_M3,
            rank_fields=["my_text_field"],
            top_n=3,
        ),
    )

asyncio.run(main())
```

---

### `AsyncIndex.search_records()`

Async alias of `AsyncIndex.search()`. Provides the same functionality with an alternative method name.

**Source:** `pinecone/db_data/index_asyncio.py:1503-1515`

**Added:** v8.1
**Deprecated:** No

---

### `AsyncIndex.query()`

Async variant of `Index.query()`. Returns a coroutine that must be awaited.

**Source:** `pinecone/db_data/index_asyncio.py:720-905`

**Added:** v8.0
**Deprecated:** No

#### Signature

```python
async def query(
    self,
    top_k: int,
    vector: list[float] | None = None,
    id: str | None = None,
    namespace: str | None = None,
    filter: FilterTypedDict | None = None,
    include_values: bool | None = None,
    include_metadata: bool | None = None,
    sparse_vector: SparseValues | SparseVectorTypedDict | None = None,
    scan_factor: float | None = None,
    max_candidates: int | None = None,
    **kwargs
) -> QueryResponse
```

#### Example

```python
import asyncio
from pinecone import Pinecone, SparseValues

async def main():
    pc = Pinecone(api_key="your-api-key")
    async_index = pc.IndexAsyncio(host="your-index-host")

    # Query with a dense vector
    response = await async_index.query(
        vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        top_k=10,
        namespace="my-namespace"
    )

    for match in response.matches:
        print(f"ID: {match.id}, Score: {match.score}")

    # Query with metadata filter and sparse vector
    response = await async_index.query(
        vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        sparse_vector=SparseValues(indices=[10, 20], values=[0.2, 0.4]),
        top_k=10,
        namespace="my-namespace",
        filter={"genre": {"$eq": "drama"}}
    )

asyncio.run(main())
```

---

## Common Patterns and Best Practices

### Pattern: Query by Vector ID

```python
response = index.query(
    id="existing-vector-id",
    top_k=10,
    namespace="my-namespace"
)
```

### Pattern: Query with Metadata Filter

```python
response = index.query(
    vector=[0.1, 0.2, 0.3],
    top_k=10,
    namespace="my-namespace",
    filter={"year": {"$gt": 2000}, "genre": "drama"}
)
```

### Pattern: Hybrid Search (Dense + Sparse)

```python
response = index.query(
    vector=[0.1, 0.2, 0.3],  # Dense embedding
    sparse_vector={"indices": [1, 5, 10], "values": [0.1, 0.2, 0.3]},  # Sparse values
    top_k=10,
    namespace="my-namespace"
)
```

### Pattern: Async Batch Querying

```python
import asyncio
from pinecone import Pinecone

async def query_multiple_namespaces():
    pc = Pinecone(api_key="your-api-key")
    async_index = pc.IndexAsyncio(host="your-index-host")

    queries = [
        {"namespace": "ns1", "vector": [0.1, 0.2, 0.3]},
        {"namespace": "ns2", "vector": [0.4, 0.5, 0.6]},
        {"namespace": "ns3", "vector": [0.7, 0.8, 0.9]},
    ]

    tasks = [
        async_index.query(**query, top_k=5)
        for query in queries
    ]

    responses = await asyncio.gather(*tasks)

    for response in responses:
        print(f"Namespace: {response.namespace}, Matches: {len(response.matches)}")

asyncio.run(query_multiple_namespaces())
```

### Pattern: Search with Reranking

```python
from pinecone import SearchQuery, SearchRerank, RerankModel

response = index.search(
    namespace="my-namespace",
    query=SearchQuery(
        inputs={"text": "query terms"},
        top_k=20,  # Fetch more results for reranking
    ),
    rerank=SearchRerank(
        model=RerankModel.Bge_Reranker_V2_M3,
        rank_fields=["text_field"],
        top_n=3,  # Return top 3 after reranking
    )
)
```

---

## Error Handling and Edge Cases

### Edge Case: Positional Arguments Not Supported

The `query()` method does not accept positional arguments. All parameters must be passed as keyword arguments.

```python
# This raises ValueError
index.query([0.1, 0.2, 0.3], 10)

# Correct usage
index.query(vector=[0.1, 0.2, 0.3], top_k=10)
```

### Edge Case: top_k Validation

`top_k` must be a positive integer (>= 1).

```python
# This raises ValueError
index.query(vector=[0.1, 0.2], top_k=0)
index.query(vector=[0.1, 0.2], top_k=-5)

# Correct usage
index.query(vector=[0.1, 0.2], top_k=1)  # OK, minimum value
```

### Edge Case: Mutual Exclusivity of vector and id

Exactly one of `vector` or `id` must be provided.

```python
# This raises ValueError (both provided)
index.query(vector=[0.1, 0.2], id="vec-1", top_k=10)

# This raises ValueError (neither provided)
index.query(top_k=10)

# Correct usage
index.query(vector=[0.1, 0.2], top_k=10)
index.query(id="vec-1", top_k=10)
```

### Edge Case: Connection Failures

If the index host is unreachable, a `ProtocolError` is raised.

```python
try:
    response = index.query(vector=[0.1, 0.2], top_k=10)
except ProtocolError as e:
    print(f"Failed to connect to index: {e}")
```

---

## Data Models

### `SearchQuery`

Represents a search query object passed to the `search()` method.

**Source:** `pinecone/db_data/dataclasses/search_query.py`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `inputs` | `dict[str, Any]` | No | v8.1 | No | Input data for the embedding model. |
| `top_k` | `int` | No | v8.1 | No | Number of results to return. |
| `filter` | `dict[str, Any] \| None` | Yes | v8.1 | No | Optional metadata filter. |
| `vector` | `SearchQueryVectorTypedDict \| SearchQueryVector \| None` | Yes | v8.1 | No | Optional vector to search with (overrides inputs). |
| `id` | `str \| None` | Yes | v8.1 | No | Optional vector ID to search with (overrides inputs). |
| `match_terms` | `dict[str, Any] \| None` | Yes | v8.1 | No | Optional matching strategy and terms for keyword matching. |

**Constructor:**

```python
from pinecone import SearchQuery

query = SearchQuery(
    inputs={"text": "search terms"},
    top_k=10,
    filter={"genre": "drama"},
    match_terms={"strategy": "all", "terms": ["term1", "term2"]}
)
```

**Methods:**

- `as_dict() -> dict[str, Any]` — Returns the SearchQuery as a dictionary (filters out None values).

---

### `SearchRerank`

Represents a reranking configuration for search results.

**Source:** `pinecone/db_data/dataclasses/search_rerank.py`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `model` | `str` | No | v8.1 | No | Name of the reranking model to use. |
| `rank_fields` | `list[str]` | No | v8.1 | No | Fields to use for reranking. |
| `top_n` | `int \| None` | Yes | v8.1 | No | Number of top results after reranking (defaults to top_k). |
| `parameters` | `dict[str, Any] \| None` | Yes | v8.1 | No | Additional model-specific parameters. |
| `query` | `str \| None` | Yes | v8.1 | No | Custom query for reranking (overrides top-level query). |

**Constructor:**

```python
from pinecone import SearchRerank, RerankModel

rerank = SearchRerank(
    model=RerankModel.Bge_Reranker_V2_M3,
    rank_fields=["text_field"],
    top_n=5
)
```

**Methods:**

- `as_dict() -> dict[str, Any]` — Returns the SearchRerank as a dictionary (filters out None values).

---

### `QueryResponse`

Represents the response from a `query()` operation.

**Source:** `pinecone/db_data/dataclasses/query_response.py`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `matches` | `list[ScoredVector]` | No | v8.0 | No | List of matched vectors with scores. Each vector contains: `id` (str), `score` (float), `values` (list[float] \| None), `metadata` (dict \| None). |
| `namespace` | `str` | No | v8.0 | No | The namespace that was queried. |
| `usage` | `Usage \| None` | Yes | v8.0 | No | Usage statistics (optional). |
| `_response_info` | `ResponseInfo` | No | v8.0 | No | Response metadata including headers (internal use). |

**Methods:**

- `as_dict() -> dict[str, Any]` — Returns the QueryResponse as a dictionary.

---

### `SparseValues`

Represents sparse vector values for hybrid search.

**Source:** `pinecone/db_data/dataclasses/sparse_values.py`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `indices` | `list[int]` | No | v8.0 | No | List of non-zero indices in the sparse vector. |
| `values` | `list[float]` | No | v8.0 | No | List of non-zero values corresponding to the indices. |

**Constructor:**

```python
from pinecone import SparseValues

sparse = SparseValues(
    indices=[0, 5, 10],
    values=[0.1, 0.2, 0.3]
)
```

**Methods:**

- `to_dict() -> SparseVectorTypedDict` — Converts to a dictionary with `indices` and `values` keys.
- `from_dict(sparse_values_dict: SparseVectorTypedDict) -> SparseValues` — Static method to create from a dictionary.

---

## Type Definitions

### `SearchQueryTypedDict`

Dictionary representation of a search query. Used as an alternative to the `SearchQuery` dataclass.

```python
from typing import TypedDict, Any

class SearchQueryTypedDict(TypedDict):
    inputs: dict[str, Any]
    top_k: int
    filter: dict[str, Any] | None
    vector: dict[str, Any] | None  # SearchQueryVectorTypedDict
    id: str | None
    match_terms: dict[str, Any] | None
```

---

### `SearchRerankTypedDict`

Dictionary representation of reranking configuration. Used as an alternative to the `SearchRerank` dataclass.

```python
from typing import TypedDict, Any
from pinecone.inference import RerankModel

class SearchRerankTypedDict(TypedDict):
    model: str | RerankModel
    rank_fields: list[str]
    top_n: int | None
    parameters: dict[str, Any] | None
    query: str | None
```

---

### `SparseVectorTypedDict`

Dictionary representation of sparse vector values.

```python
from typing import TypedDict

class SparseVectorTypedDict(TypedDict):
    indices: list[int]
    values: list[float]
```

---

### `FilterTypedDict`

Represents a metadata filter for queries. Uses MongoDB-like query syntax.

```python
FilterTypedDict = SimpleFilter | AndFilter | OrFilter

# Where:
# SimpleFilter: exact match or comparison filter (e.g., {"genre": "drama"})
# AndFilter: {"$and": [condition1, condition2, ...]}
# OrFilter: {"$or": [condition1, condition2, ...]}
```

---

## Version History

| Version | Changes |
|---------|---------|
| v8.1.0 | Added `search()` and `search_records()` methods for embedding-based search with integrated inference. |
| v8.0.0 | Initial release of `query()` method for vector-based search. |
