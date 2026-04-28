# Working with Collections

Collections are read-only snapshots of pod indexes. Use them to back up index data,
duplicate an index, or restore a known-good state. Collections are only supported for
pod-based indexes — serverless indexes use backups instead.

## Create a collection

Pass the name of the pod index you want to snapshot:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

collection = pc.collections.create(name="snap-2025-01", source="my-pod-index")
print(collection.status)   # "Initializing" immediately after creation
```

The collection transitions through ``Initializing`` → ``Ready`` when the snapshot is
complete. ``create`` returns immediately without polling; check status with ``describe``.

## List collections

``list`` returns a :class:`~pinecone.models.collections.list.CollectionList` you can
iterate or call ``.names()`` on:

```python
for col in pc.collections.list():
    print(col.name, col.status)
```

```python
names = pc.collections.list().names()
print(names)   # e.g. ["snap-2025-01", "archive-q3"]
```

## Describe a collection

``describe`` returns a :class:`~pinecone.models.collections.model.CollectionModel` with
detailed information:

```python
col = pc.collections.describe("snap-2025-01")
print(col.name)          # "snap-2025-01"
print(col.status)        # "Ready"
print(col.dimension)     # vector dimension
print(col.vector_count)  # number of vectors stored
print(col.size)          # size in bytes
print(col.environment)   # cloud environment
```

Poll until ready after creation:

```python
import time

while True:
    col = pc.collections.describe("snap-2025-01")
    if col.status == "Ready":
        break
    time.sleep(5)
```

## Delete a collection

```python
pc.collections.delete("snap-2025-01")
```

``delete`` raises :exc:`~pinecone.exceptions.NotFoundError` if the collection does not
exist.

## Create an index from a collection

Pass ``source_collection`` inside a :class:`~pinecone.PodSpec` to restore collection
data into a new pod index:

```python
from pinecone import Pinecone
from pinecone.models.indexes.specs import PodSpec

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="restored-index",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east-1-aws",
        pod_type="p1.x1",
        source_collection="snap-2025-01",
    ),
)
```

The new index is pre-populated with all vectors from the collection snapshot.
