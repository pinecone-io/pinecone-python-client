# Upserting and querying vectors

This page serves indexes created with the deprecated top-level `dimension=`/`metric=`.
On a schema-based (document) index the server rejects these methods; write and search
through {attr}`~pinecone.index.Index.documents` instead (see [Choosing an
interface](../../getting-started/quickstart.md#choosing-an-interface)).

Get an {class}`~pinecone.index.Index` client with {meth}`~pinecone.Pinecone.index`:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("movie-recommendations")
```

Vectors on this page are written as three floats so they fit the page. Pass your index's
full dimension; the server rejects a vector of the wrong length.


## Upsert vectors

{meth}`~pinecone.index.Index.upsert` inserts vectors, overwriting any existing vector with
the same ID.

### The three vector shapes

`vectors` accepts a tuple, a dict, or a {class}`~pinecone.models.vectors.vector.Vector`, and
the three can be mixed within one call:

```python
from pinecone import Vector

response = index.upsert(
    vectors=[
        # (id, values) or (id, values, metadata)
        ("movie-001", [0.012, -0.087, 0.153]),
        ("movie-002", [0.045, 0.021, -0.064], {"genre": "comedy", "year": 2022}),
        # a dict, keys drawn from id / values / sparse_values / metadata
        {"id": "movie-003", "values": [0.091, -0.032, 0.178]},
        {
            "id": "movie-004",
            "values": [0.020, 0.030, 0.040],
            "sparse_values": {"indices": [7, 21], "values": [0.4, 0.6]},
            "metadata": {"genre": "drama"},
        },
        # a Vector object, which your editor can check
        Vector(id="movie-005", values=[0.063, 0.011, -0.022]),
    ]
)
print(response.upserted_count)
```

The client rejects a dict with any other key before sending anything, so put your own
fields in `metadata`. Metadata values must be a string, a number, a boolean, or a list of
strings; anything else raises {exc}`~pinecone.errors.exceptions.PineconeTypeError` before
the request is sent, rather than failing the whole request at the server.

### Upsert into a namespace

Pass `namespace` to target a specific partition:

```python
index.upsert(
    vectors=[("movie-001", [0.012, -0.087, 0.153])],
    namespace="movies-en",
)
```

Omitting `namespace` writes to the default namespace. Namespaces are isolated, so a
vector written to `"movies-en"` is invisible to a query that does not name it. See
{doc}`/how-to/vectors/namespaces`.

### Large datasets

A single upsert request is capped on both vector count and encoded size. Wide vectors or
heavy metadata usually reach the size cap first, so a vector count that worked for one
dataset can be rejected for another. Pass `batch_size` to split the upload into requests
that stay under both caps, and lower it if a request is rejected for size. The client
sends up to `max_concurrency` batches at once and retries transient failures per batch.

```python
response = index.upsert(
    vectors=large_list,
    batch_size=200,        # vectors per request
    max_concurrency=8,     # most requests in flight at once
    total_timeout=600.0,   # deadline in seconds for the whole call
)
print(response.upserted_count)         # vectors that landed
print(response.total_item_count)       # vectors submitted
```

`timeout` bounds one attempt of one batch, so use `total_timeout` to bound the whole
call. For what each knob does and how the client paces batches, see [How bulk ingest
behaves](../../guides/bulk-ingest.md).

For DataFrame input, {meth}`~pinecone.index.Index.upsert_from_dataframe` batches the same
way (see {doc}`/how-to/vectors/upsert-from-dataframe`). For millions of vectors in cloud
storage, use {meth}`~pinecone.index.Index.start_import` (see
{doc}`/how-to/vectors/bulk-import`).

### Handling partial failures

Without `batch_size`, a failed upsert raises. With `batch_size`, a failed batch does not
raise: the returned {class}`~pinecone.models.vectors.responses.UpsertResponse` lists each
failed batch in `errors`, and `failed_items` collects the items from every failed batch as
dicts you can pass straight back to `upsert`.

The client has already retried transient failures by the time a batch lands in `errors`
(see {doc}`/guides/retries`), so check each error's `retryable` flag before resending.
`False` marks a deterministic rejection, such as bad input or bad credentials, that fails
the same way every time:

```python
response = index.upsert(vectors=large_list, batch_size=200)

if response.has_errors:
    print(f"{response.failed_item_count} of {response.total_item_count} vectors failed")
    for err in response.errors:
        print(f"  batch {err.batch_index}: {err.error_message}")

    if all(err.retryable for err in response.errors):
        retry = index.upsert(vectors=response.failed_items, batch_size=200)
```

A batch fails as a whole, so one bad vector fails every vector in its batch. To find the
bad vectors, resend the failures one per request; only the bad ones fail again:

```python
narrow = index.upsert(vectors=response.failed_items, batch_size=1)
# narrow.failed_items now holds only the vectors that were rejected
```


## Query for nearest neighbors

### `query` or `search`: pick one

Two search methods exist, and which one your index supports is fixed when the index is
created.

- {meth}`~pinecone.index.Index.query`: **you supply the vector.** Nothing is embedded for
  you. This is the method for an index created with top-level `dimension` and `metric`,
  and it is what the rest of this page uses.
- {meth}`~pinecone.index.Index.search`: **you supply text**, and the index's embedding
  model turns it into the query vector on the server. This needs a legacy integrated
  index, created with {meth}`~pinecone.client.indexes.Indexes.create_for_model`. See
  {doc}`/how-to/integrated-records`.

Both accept the same `filter` dict, so [Filter by metadata](#filter-by-metadata) below
applies to `search` too.

### Running a query

`query` returns the `top_k` vectors closest to the vector you give it:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153],
    top_k=10,
)
for match in response.matches:
    print(match.id, match.score)
```

Each element of `response.matches` is a {class}`~pinecone.models.vectors.vector.ScoredVector`,
ordered most similar first. A query that matches nothing returns an empty `matches` rather
than raising.

To query by a vector the index already holds, pass its ID as `index.query(id=...,
top_k=10)` instead of `vector`. An `id` cannot be combined with `vector` or
`sparse_vector`.

### Include values or metadata in results

Only `id` and `score` are always populated. `values` and `metadata` are left out of the
response unless you ask for them:

:::{warning}
Without `include_values=True`, `match.values` is an **empty list**, not the stored
vector. Without `include_metadata=True`, `match.metadata` is **`None`**, so
`match.metadata["genre"]` raises `TypeError` even for a vector that has metadata stored,
and even when you filtered on that field.
:::

```python
response = index.query(
    vector=[0.012, -0.087, 0.153],
    top_k=10,
    include_values=True,
    include_metadata=True,
)
for match in response.matches:
    print(match.id, match.score, match.values[:3], match.metadata)
```

### Filter by metadata

A `filter` is a plain dict:

```python
response = index.query(
    vector=[0.012, -0.087, 0.153],
    top_k=5,
    filter={"genre": {"$eq": "action"}, "year": {"$gte": 2020}},
    include_metadata=True,
)
```

The operators are:

| Operator | Takes | Means |
|---|---|---|
| `$eq` / `$ne` | a string, number, or boolean | equal / not equal |
| `$gt` / `$gte` / `$lt` / `$lte` | a number | ordering comparison |
| `$in` / `$nin` | a list | is / is not one of |
| `$exists` | a boolean | the field is present |
| `$and` / `$or` | a list of clauses | combine clauses |

Naming a field once with several operators, as in `{"year": {"$gte": 2020, "$lte":
2024}}`, is an implicit `$and`. The same `filter` argument selects records for
{meth}`~pinecone.index.Index.update` and {meth}`~pinecone.index.Index.delete`.

### Building filters in code

Reach for {class}`~pinecone.utils.filter_builder.Field` when your code *assembles* a filter rather than
writing one out: it gives one method per operator, so your editor checks the operator
names instead of you typing them into a string, and `&` / `|` compose clauses built in
different places. `==` and `!=` build `$eq` and `$ne`; `.gt()` / `.gte()` / `.lt()` /
`.lte()` are numeric only; `.is_in()` / `.not_in()` take a list; `.exists()` takes
nothing. Each returns a {class}`~pinecone.utils.filter_builder.Condition`, and
`.to_dict()` produces the dict to pass as `filter`:

```python
from pinecone import Field

condition = (Field("genre") == "action") & Field("year").gte(2020)
print(condition.to_dict())
# {'$and': [{'genre': {'$eq': 'action'}}, {'year': {'$gte': 2020}}]}

response = index.query(
    vector=[0.012, -0.087, 0.153],
    top_k=5,
    filter=condition.to_dict(),
    include_metadata=True,
)
```

Both routes produce the same filter, so mixing them is fine. Because `==` is overloaded
to build a filter rather than answer a question, a `Field` never compares equal to
anything and cannot be used as a dict key or a set member.



## Fetch vectors by ID

{meth}`~pinecone.index.Index.fetch` retrieves stored vectors by their IDs:

```python
wanted = ["movie-001", "movie-002"]
response = index.fetch(ids=wanted)
for vid, vec in response.vectors.items():
    print(vid, vec.values[:3], vec.metadata)
print("missing:", set(wanted) - set(response.vectors))
```

`response.vectors` maps each ID to a {class}`~pinecone.models.vectors.vector.Vector`. An ID
the namespace does not hold is absent from the result rather than raising, so compare the
keys you got back against the ones you asked for.


## Update a vector

{meth}`~pinecone.index.Index.update` patches one vector's dense values, sparse values, or
metadata. Fields you do not name keep the values they had, so `set_metadata={"year":
2021}` leaves every other metadata key in place. Give exactly one selector, `id` or
`filter`; an update by `filter` can only change metadata.

Update dense values by ID:

```python
index.update(id="movie-001", values=[0.099, -0.045, 0.210])
```

Update metadata without changing values:

```python
index.update(id="movie-001", set_metadata={"rating": 4.5, "genre": "thriller"})
```

Update metadata on every vector matching a filter, checking the reach first with
`dry_run=True`, which reports the match count without changing anything:

```python
preview = index.update(
    filter={"genre": {"$eq": "drama"}},
    set_metadata={"category": "classic"},
    dry_run=True,
)
print(preview.matched_records)

index.update(
    filter={"genre": {"$eq": "drama"}},
    set_metadata={"category": "classic"},
)
```

An update applies asynchronously, so a read straight afterwards can still see the old
value.


## Delete vectors

{meth}`~pinecone.index.Index.delete` removes vectors from a namespace. Specify exactly one
of `ids`, `delete_all`, or `filter`. Deletes are irreversible, and IDs the namespace does
not hold are ignored rather than reported, so a successful call is not evidence anything
was deleted.

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

{meth}`~pinecone.index.Index.describe_index_stats` returns aggregate counts and
per-namespace summaries:

```python
stats = index.describe_index_stats()
print(stats.total_vector_count, stats.dimension, stats.index_fullness)

for namespace, summary in stats.namespaces.items():
    print(namespace, summary.vector_count)
```

The counts always cover the whole index, and the default namespace is listed as
`"__default__"`. The server rejects a non-empty `filter`, so there is no way to count only
the vectors that match a metadata filter.


## See also

- {doc}`/how-to/vectors/namespaces`: working with namespaces
- {doc}`/how-to/vectors/bulk-import`: bulk importing from cloud storage
- {class}`~pinecone.index.Index`: full data-plane client reference
- {doc}`/how-to/integrated-records`: `search` on a legacy integrated index
- {doc}`/guides/performance`: choosing a `batch_size` and a `max_concurrency`
