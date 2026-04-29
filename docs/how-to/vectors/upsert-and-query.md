# Upserting and Querying Vectors

Use the {class}`~pinecone.Index` client to insert and retrieve vectors from a Pinecone index.
Get an index client via {meth}`~pinecone.Pinecone.index`:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("movie-recommendations")
```


## Upsert vectors

{meth}`~pinecone.Index.upsert` inserts vectors or overwrites existing ones with the same ID.

Pass a list of tuples `(id, values)` or `(id, values, metadata)`:

```python
index.upsert(
    vectors=[
        ("movie-001", [0.012, -0.087, 0.153, ...]),
        ("movie-002", [0.045,  0.021, -0.064, ...]),
    ]
)
```

### Using Vector objects

{class}`~pinecone.Vector` objects support metadata and sparse values:

```python
from pinecone import Vector

response = index.upsert(
    vectors=[
        Vector(id="movie-001", values=[0.012, -0.087, 0.153, ...]),
        Vector(
            id="movie-002",
            values=[0.045, 0.021, -0.064, ...],
            metadata={"genre": "comedy", "year": 2022},
        ),
    ]
)
print(response.upserted_count)  # 2
```

`upsert` returns a {class}`~pinecone.models.UpsertResponse` with `upserted_count`.

### Upsert into a namespace

Pass `namespace` to target a specific partition:

```python
index.upsert(
    vectors=[("movie-001", [0.012, -0.087, 0.153, ...])],
    namespace="movies-en",
)
```

The default namespace is `""`.

### Large datasets

`upsert` sends all vectors in a single request. For large datasets, batch your calls
(100–500 vectors per batch) or use {meth}`~pinecone.Index.upsert_from_dataframe` for
DataFrame input with automatic batching. For millions of vectors, consider
{meth}`~pinecone.Index.start_import` to load from cloud storage.


## Query for nearest neighbors

{meth}`~pinecone.Index.query` returns the `top_k` closest vectors to a query vector:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=10,
)
for match in response.matches:
    print(match.id, match.score)
```

Each element of `response.matches` is a {class}`~pinecone.models.ScoredVector` with
`id`, `score`, `values`, `metadata`, and `sparse_values` fields. Results are ordered from
most similar to least similar.

### Include values or metadata in results

By default, `values` and `metadata` are omitted from matches to reduce payload size.
Enable them explicitly:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=10,
    include_values=True,
    include_metadata=True,
)
for match in response.matches:
    print(match.id, match.score, match.metadata)
```

### Filter by metadata

Pass a `filter` expression to restrict results to vectors whose metadata satisfies the condition:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=5,
    filter={"genre": {"$eq": "action"}, "year": {"$gte": 2020}},
    include_metadata=True,
)
```

### Using the Field filter builder

{class}`~pinecone.Field` provides a Python-native API for building filter expressions.
The `==`, `!=`, `&`, and `|` operators and `.gt()` / `.gte()` / `.lt()` / `.lte()` /
`.is_in()` / `.not_in()` methods return a {class}`~pinecone.utils.filter_builder.Condition`
object. Pass it to `filter` via `.to_dict()`:

```python
from pinecone import Field

condition = (Field("genre") == "action") & Field("year").gte(2020)

response = index.query(
    vector=[0.012, -0.087, 0.153, ...],
    top_k=5,
    filter=condition.to_dict(),
    include_metadata=True,
)
```

`FilterBuilder` is an alias for `Field` and can be used interchangeably.


## Fetch vectors by ID

{meth}`~pinecone.Index.fetch` retrieves stored vectors by their IDs:

```python
response = index.fetch(ids=["movie-001", "movie-002"])
for vid, vec in response.vectors.items():
    print(vid, vec.values[:3])
```

`response.vectors` is a `dict[str, Vector]`. IDs that do not exist are omitted rather than
raising an error.


## Update a vector

{meth}`~pinecone.Index.update` replaces a vector's dense values, sparse values, or metadata.

Update dense values by ID:

```python
index.update(id="movie-001", values=[0.099, -0.045, 0.210, ...])
```

Update metadata without changing values:

```python
index.update(id="movie-001", set_metadata={"rating": 4.5, "genre": "thriller"})
```

Bulk-update metadata for all vectors matching a filter:

```python
index.update(
    filter={"genre": {"$eq": "drama"}},
    set_metadata={"category": "classic"},
)
```


## Delete vectors

{meth}`~pinecone.Index.delete` removes vectors from a namespace. Specify exactly one of
`ids`, `delete_all`, or `filter`.

Delete by ID:

```python
index.delete(ids=["movie-001", "movie-002"])
```

Delete all vectors in a namespace:

```python
index.delete(delete_all=True, namespace="movies-deprecated")
```

Delete by metadata filter:

```python
index.delete(filter={"year": {"$lte": 2000}})
```


## Inspect index stats

{meth}`~pinecone.Index.describe_index_stats` returns aggregate counts and
per-namespace summaries:

```python
stats = index.describe_index_stats()
print(stats.total_vector_count)
print(stats.dimension)
print(stats.index_fullness)     # fraction 0.0–1.0

for namespace, summary in stats.namespaces.items():
    print(namespace, summary.vector_count)
```

Pass a `filter` to count only matching vectors:

```python
stats = index.describe_index_stats(filter={"genre": {"$eq": "action"}})
print(stats.total_vector_count)
```


## See also

- {doc}`/how-to/vectors/namespaces` — working with namespaces
- {doc}`/how-to/vectors/bulk-import` — bulk importing from cloud storage
- {class}`~pinecone.Index` — full data plane client reference
- {class}`~pinecone.models.QueryResponse` — query response model
- {class}`~pinecone.models.ScoredVector` — individual match in query results
