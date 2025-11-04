# Vectors

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

## Fetch vectors by metadata

The following example fetches vectors by metadata filter.

```python
import os
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

# Find your index host by calling describe_index
# through the Pinecone web console
index = pc.Index(host=os.environ.get('INDEX_HOST'))

# Fetch vectors matching a metadata filter
fetch_response = index.fetch_by_metadata(
    filter={"genre": {"$in": ["comedy", "drama"]}, "year": {"$eq": 2019}},
    namespace="example-namespace",
    limit=50
)

# Iterate over the fetched vectors
for vec_id, vector in fetch_response.vectors.items():
    print(f"Vector ID: {vector.id}")
    print(f"Metadata: {vector.metadata}")

# Handle pagination if there are more results
if fetch_response.pagination:
    next_page = index.fetch_by_metadata(
        filter={"genre": {"$in": ["comedy", "drama"]}, "year": {"$eq": 2019}},
        namespace="example-namespace",
        pagination_token=fetch_response.pagination.next
    )
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
