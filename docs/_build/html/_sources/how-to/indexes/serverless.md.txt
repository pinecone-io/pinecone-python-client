# Working with Serverless Indexes

Serverless indexes scale automatically — you pay for storage and queries without managing
infrastructure. Pinecone handles capacity, replication, and availability.

## Create a serverless index

Pass a :class:`~pinecone.ServerlessSpec` with a cloud provider and region:

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

`create` polls until the index is ready by default. Pass `timeout=-1` to return immediately
without waiting.

### Supported clouds and regions

Use the :class:`~pinecone.CloudProvider`, :class:`~pinecone.AwsRegion`,
:class:`~pinecone.GcpRegion`, and :class:`~pinecone.AzureRegion` enums for
tab-completion and typo safety:

```python
from pinecone import Pinecone, ServerlessSpec
from pinecone.models.enums import AwsRegion, CloudProvider

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1,
    ),
)
```

**AWS:** ``us-east-1``, ``us-west-2``, ``eu-west-1``

**GCP:** ``us-central1``, ``europe-west4``

**Azure:** ``eastus2``

### Enable deletion protection

Add `deletion_protection="enabled"` to prevent accidental deletes:

```python
from pinecone import Pinecone, ServerlessSpec
from pinecone.models.enums import DeletionProtection

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    deletion_protection=DeletionProtection.ENABLED,
)
```


## Check index status

`describe` returns an :class:`~pinecone.models.IndexModel` with the current state:

```python
desc = pc.indexes.describe("product-search")
print(desc.status.state)   # e.g. "Ready"
print(desc.status.ready)   # True when ready to accept requests
```

Poll manually when you passed `timeout=-1` to `create`:

```python
import time

while not pc.indexes.describe("product-search").status.ready:
    time.sleep(5)
```


## List indexes

`list` returns an :class:`~pinecone.models.IndexList` you can iterate, slice, or call
`.names()` on:

```python
for idx in pc.indexes.list():
    print(idx.name, idx.status.state)

# Just the names
print(pc.indexes.list().names())
```


## Describe an index

```python
idx = pc.indexes.describe("product-search")
print(idx.name)
print(idx.dimension)
print(idx.metric)
print(idx.spec.serverless.cloud)
print(idx.spec.serverless.region)
```


## Delete an index

```python
pc.indexes.delete("product-search")
```

`delete` polls until the index is gone. Pass `timeout=-1` to return immediately.

If deletion protection is enabled, disable it first:

```python
pc.indexes.configure("product-search", deletion_protection="disabled")
pc.indexes.delete("product-search")
```


## See also

- :class:`~pinecone.models.IndexModel` — full index response model
- :class:`~pinecone.models.IndexList` — list response model
- :doc:`/how-to/indexes/pod` — pod-based index management
- :doc:`/how-to/indexes/backups-and-restore` — create and restore backups
