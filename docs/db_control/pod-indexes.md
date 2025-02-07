# Pod Indexes

This page talks about working with pod-based indexes with python. For an overview of the
most important concepts, please see [Understanding pod-based indexes](https://docs.pinecone.io/guides/indexes/pods/understanding-pod-based-indexes)

## Create a pod index

The following example creates an index without a metadata
configuration. By default, Pinecone indexes all metadata.

Many of these fields accept string literals if you know
the values you want to use, but enum objects such as
`PodIndexEnvironment`, `PodType`, `Metric` and more can
help you discover available options.

This examples shows some optional properties, [tags and deletion protection](shared-index-configs.md), in use.

```python
from pinecone import (
    Pinecone,
    PodSpec,
    Metric,
    DeletionProtection,
    PodType,
    PodIndexEnvironment
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
index_config = pc.create_index(
    name="example-index",
    dimension=1536,
    metric=Metric.COSINE,
    deletion_protection=DeletionProtection.ENABLED,
    spec=PodSpec(
        environment=PodIndexEnvironment.EU_WEST1_GCP,
        pod_type=PodType.P1_X1
    ),
    tags={
        "environment": "production",
        "app": "chat-support"
    }
)
print(index_config)
# {
#     "name": "example-index",
#     "metric": "cosine",
#     "host": "example-index-dojoi3u.svc.eu-west1-gcp.pinecone.io",
#     "spec": {
#         "pod": {
#             "environment": "eu-west1-gcp",
#             "pod_type": "p1.x1",
#             "replicas": 1,
#             "shards": 1,
#             "pods": 1
#         }
#     },
#     "status": {
#         "ready": true,
#         "state": "Ready"
#     },
#     "vector_type": "dense",
#     "dimension": 1536,
#     "deletion_protection": "enabled",
#     "tags": {
#         "app": "chat-support",
#         "environment": "production"
#     }
# }
```

This create command will block for a few moments (seconds to minutes) while
pods are being deployed. If you would prefer for the function to return immediately
instead of waiting for the index to be ready for use, you can add a
`timeout=-1` argument to your `create_index` call.


## Optional spec configurations when creating pod indexes

Pod indexes support many optional configuration fields through
the spec object. For example, if your workload requires a more powerful
pod type or additional replicas, those would be indicated using keyword
arguments to the `PodSpec` object.

Also, if you have [high-cardinality metadata](https://docs.pinecone.io/guides/data/understanding-metadata#manage-high-cardinality-in-pod-based-indexes)
stored with your vectors, you can significantly improve your memory
performance by telling pinecone which fields you plan to use for filtering.
These settings are conveyed with an optional `metadata_config` keyword param.

The following example creates an index that only indexes the "color"
metadata field for queries with filtering; with this metadata configuration,
queries against the index cannot filter based on any other metadata field.

```python
from pinecone import Pinecone, PodSpec, PodType, PodIndexEnvironment

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name="example-index-2",
    dimension=1536,
    spec=PodSpec(
        environment=PodIndexEnvironment.EU_WEST1_GCP,
        pod_type=PodType.P1_X1,
        metadata_config={
            "indexed": ["color"]
        },
        replicas=2
    )
)
```


## Creating a pod-based index from a collection

For more info on collections, see [Collections](./collections.md)

```
from pinecone import Pinecone, PodSpec, PodIndexEnvironment

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name=index_name,
    dimension=1536,
    metric=metric,
    spec=PodSpec(
        environment=PodIndexEnvironment.US_WEST1_GCP,
        source_collection="name-of-collection"
    ),
)
```

## Scaling your pod-based index with `configure_index`

Please see this page [Scaling pod-based indexes](https://docs.pinecone.io/guides/indexes/pods/scale-pod-based-indexes) for
an introduction to core concepts related to scaling Pinecone indexes.

If you wish to scale horizontally with `replicas` or veritcally with `pod_type`, both of those fields can be passed
to `configure_index` to make changes to an existing index. Changes are not instantaneous; call `describe_index` to
see whether the configuration change has been completed.

```python
from pinecone import Pinecone, PodType

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.configure_index(
    name="example-index",
    replicas=4,
    pod_type=PodType.P1_X2
)
```

# Configuring, listing, describing, and deleting

See [shared index actions](shared-index-actions.md) to learn about how to manage the lifecycle of your index after it is created.
