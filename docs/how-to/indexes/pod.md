# Working with pod-based indexes

Pod-based indexes run on dedicated infrastructure pods. A pod type and pod count determine
the index's throughput, latency, and storage capacity. They predate serverless indexes;
reach for a [serverless index](serverless.md) unless you have a reason not to.

```{important}
API version `2026-07`, the version this SDK targets, does not create pod-based indexes. The
server rejects a {meth}`~pinecone.client.indexes.Indexes.create` call that carries a pod
deployment (`deployment={"deployment_type": "pod", ...}` or the deprecated
`spec=PodSpec(...)`) with a 400 {exc}`~pinecone.errors.exceptions.ApiError`.

Every other operation still works on pod indexes you already have: `describe()` and `list()`
decode them, and `configure()` and `delete()` act on them. This page covers those
operations. See {ref}`pod-collections` in the v10 migration guide for the header pin that
reaches an earlier API version.
```

For a new index, create a serverless index instead:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

pc.indexes.create(
    name="product-search",
    schema={"fields": {"your_vector": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
    deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
)
```

See [serverless indexes](serverless.md) for the managed deployment options.


## Describe a pod index

The `deployment` field decodes as a
{class}`~pinecone.models.indexes.deployment.PodDeployment`:

```python
idx = pc.indexes.describe("legacy-recommender")
print(idx.deployment.environment)  # e.g. "us-east-1-aws"
print(idx.deployment.pod_type)     # e.g. "p1.x1"
print(idx.deployment.replicas)
print(idx.deployment.shards)
```

The environment was fixed when the index was created. A pod index sizes reads with
`pod_type`, `replicas`, and `shards`, so it has no `read_capacity`.

### Pod types

| Pod type | Optimized for |
|---|---|
| ``s1.x1``, ``s1.x2``, ``s1.x4``, ``s1.x8`` | Storage, at lower query throughput |
| ``p1.x1``, ``p1.x2``, ``p1.x4``, ``p1.x8`` | Balanced performance |
| ``p2.x1``, ``p2.x2``, ``p2.x4``, ``p2.x8`` | Query throughput, at lower storage capacity |

The family before the dot picks what the pod is optimized for; the ``xN`` after it is the
size multiplier. The {class}`~pinecone.models.enums.PodType` enum names these values.

### Replicas and shards

Replicas duplicate the index for higher availability and query throughput. Shards split the
index's data across pods, which decides how much data fits. The total pod count is replicas
times shards.

You can change `replicas` and `pod_type` on a running index. `shards` is fixed once the
index exists, so an index that has outgrown its shard count has to be rebuilt.


## Scale a pod index

Change replicas, pod type, or both with
{meth}`~pinecone.client.indexes.Indexes.configure`:

```python
from pinecone.models.enums import PodType

pc.indexes.configure("legacy-recommender", deployment={"replicas": 4, "pod_type": PodType.P1_X2})
```

The call returns while the change is still in progress. Poll
{meth}`~pinecone.client.indexes.Indexes.describe` and read `status.state`, which reports a
scaling state such as `"ScalingUp"` until the change is done.

`deployment` here carries only `replicas` and `pod_type`. Deployment type, cloud, region, and
environment can't change after creation, and a `deployment` dict that includes
`deployment_type` raises {exc}`~pinecone.errors.exceptions.PineconeValueError`.

The deprecated `replicas=` and `pod_type=` keyword arguments still work, but passing either
together with `deployment=` raises {exc}`~pinecone.errors.exceptions.PineconeValueError`.


## Collections

A collection is a static snapshot of a pod index's vector data. Collections that already
exist can still be listed, described, and deleted; see [collections](../collections.md).
For snapshot and restore on serverless indexes, use [backups](backups-and-restore.md).


## Delete a pod index

```python
pc.indexes.delete("legacy-recommender")
```

{meth}`~pinecone.client.indexes.Indexes.delete` polls until the index is gone. Pass
`timeout=-1` to return as soon as the request is accepted.

If deletion protection is enabled, the server refuses the delete with
{exc}`~pinecone.errors.exceptions.ForbiddenError`. Disable it first:

```python
pc.indexes.configure("legacy-recommender", deletion_protection="disabled")
pc.indexes.delete("legacy-recommender")
```

The server also refuses the delete while a collection from this index is still being
created or deleted. Wait for that collection to finish, then retry.


## See also

- {class}`~pinecone.models.indexes.index.IndexModel`: full index response model
- {class}`~pinecone.models.indexes.deployment.PodDeployment`: the pod deployment on a describe
- {class}`~pinecone.models.indexes.specs.PodSpec`: deprecated `create()` sugar for `deployment=`
