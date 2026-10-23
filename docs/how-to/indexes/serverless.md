# Working with serverless indexes

Serverless indexes scale automatically: you pay for storage and queries, and Pinecone
handles capacity, replication, and availability.

## Create a serverless index

{meth}`~pinecone.client.indexes.Indexes.create` takes the index's fields as a `schema`
and its cloud and region as a `deployment`:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    schema={"fields": {"your_vector": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

`"deployment_type": "managed"` is what makes an index serverless. Omit `deployment` and
the server creates a managed index on AWS `us-east-1`.

Only fields that get searched go in the schema: `dense_vector`, `sparse_vector`, or
`string` with a `full_text_search`, `embed`, or `sparse_embed` config (the last two
enable [integrated embedding](../../guides/concepts.md#integrated-embedding)). Leave out
metadata you only filter on: put it on the records you upsert and the server indexes it
for filtering automatically. The server rejects a schema that declares a metadata-only
field. The schema can't change after the index exists, so a hybrid index has to declare
its `sparse_vector` field at create time.

`create` polls until the index is ready, with no upper time bound. Pass a positive
`timeout` to bound the wait (it raises
{exc}`~pinecone.errors.exceptions.PineconeTimeoutError` if the index still isn't ready),
or `timeout=-1` to return as soon as the server accepts the request.

`dimension=`, `metric=`, `vector_type=`, and `spec=` are deprecated keyword arguments
that the SDK translates into `schema=` and `deployment=`. Combining one with the argument
it translates to raises {exc}`~pinecone.errors.exceptions.PineconeValueError`. See the
[v10 migration guide](../../migration/v10-migration.md) to move code off them.

### Clouds and regions

{class}`~pinecone.models.enums.CloudProvider` and the per-cloud region enums
{class}`~pinecone.models.enums.AwsRegion`, {class}`~pinecone.models.enums.GcpRegion`, and
{class}`~pinecone.models.enums.AzureRegion` give you tab-completion and typo safety:

```python
from pinecone import AwsRegion, CloudProvider, Pinecone

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    schema={"fields": {"your_vector": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={
        "deployment_type": "managed",
        "cloud": CloudProvider.AWS,
        "region": AwsRegion.US_EAST_1,
    },
)
```

The SDK sends `cloud` and `region` as plain strings without checking them, so a region
Pinecone adds before an enum member names it works as a string literal, and a typo comes
back as a server error.

### Enable deletion protection

Pass `deletion_protection="enabled"` to keep the index from being deleted by accident:

```python
pc.indexes.create(
    name="product-search",
    schema={"fields": {"your_vector": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    deletion_protection="enabled",
)
```


## Choose read capacity

`read_capacity` decides how the index serves reads and how they are billed. Omit it and
the index uses on-demand capacity, billed per operation. Dedicated capacity provisions
read nodes that you size yourself:

```python
pc.indexes.create(
    name="product-search",
    schema={"fields": {"your_vector": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    read_capacity={
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {"shards": 2, "replicas": 2},
        },
    },
)
```

`node_type` is `"b1"` or `"t1"`, and `manual` sets the `shards` and `replicas` to
provision. Setting `replicas` to `0` disables the index, which cuts cost while a workload
is paused without dropping the data.

{meth}`~pinecone.client.indexes.Indexes.configure` takes the same argument to change an
existing index. The change applies asynchronously, so `configure` returns while it is
still in flight:

```python
pc.indexes.configure("product-search", read_capacity={"mode": "OnDemand"})
```

`describe(...).read_capacity` is a
{class}`~pinecone.models.indexes.read_capacity.ReadCapacityOnDemandResponse` or a
{class}`~pinecone.models.indexes.read_capacity.ReadCapacityDedicatedResponse`, or `None`
when the response carries none. Neither class has a `mode` attribute; branch on the type:

```python
from pinecone import ReadCapacityDedicatedResponse

rc = pc.indexes.describe("product-search").read_capacity
if rc is not None:
    print(rc.status.state)  # 'Ready'
    if isinstance(rc, ReadCapacityDedicatedResponse):
        print(rc.dedicated.node_type, rc.dedicated.manual.shards)
```

`status.state` shows an asynchronous change in progress: `"Scaling"` after a replica or
shard change, `"Migrating"` while moving to another node type, `"Ready"` once settled,
and `"Error"` with the reason in `status.error_message`. `status.current_shards` and
`status.current_replicas` report what is actually running.

Pod-based indexes don't take `read_capacity`; they size reads with pod type, replicas,
and shards (see [pod-based indexes](pod.md)).


## Check index status

{meth}`~pinecone.client.indexes.Indexes.describe` returns an
{class}`~pinecone.models.indexes.index.IndexModel`. If you passed `timeout=-1` to
`create`, poll `status.ready` until the index can serve requests:

```python
import time

while not pc.indexes.describe("product-search").status.ready:
    time.sleep(5)
```


## List indexes

{meth}`~pinecone.client.indexes.Indexes.list` returns a
{class}`~pinecone.models.pagination.Paginator` that yields every index in the project:

```python
names = [idx.name for idx in pc.indexes.list()]
```


## Describe an index

The dimension and metric live on each vector field in `schema.fields`, not on the index:

```python
idx = pc.indexes.describe("product-search")
print(idx.host, idx.deployment.cloud, idx.deployment.region)
vector_field = idx.schema.fields["your_vector"]
print(vector_field.dimension, vector_field.metric)
```

`idx.dimension` and `idx.metric` are deprecated. They resolve only when the schema has a
single dense vector field, and raise `AttributeError` naming the fields when it has
several.


## Delete an index

```python
pc.indexes.delete("product-search")
```

{meth}`~pinecone.client.indexes.Indexes.delete` polls until the index is gone, with no
upper time bound. Pass a positive `timeout` to bound the wait, or `timeout=-1` to return
while the server is still tearing the index down.

The server refuses to delete an index with deletion protection enabled, raising
{exc}`~pinecone.errors.exceptions.ForbiddenError`. Disable it first:

```python
pc.indexes.configure("product-search", deletion_protection="disabled")
pc.indexes.delete("product-search")
```


## See also

- [Backups and restore](backups-and-restore.md): back up a serverless index and restore it
- [Concepts](../../guides/concepts.md): how deployments, schemas, and namespaces fit together
