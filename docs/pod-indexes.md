# Pod Indexes

### Create a pod index

The following example creates an index without a metadata
configuration. By default, Pinecone indexes all metadata.

Many of these fields accept string literals if you know
the values you want to use, but enum objects such as
`PodIndexEnvironment`, `PodType`, `Metric` and more can
help you discover available options.

```python
from pinecone import (
    Pinecone,
    PodSpec,
    Metric,
    PodType,
    PodIndexEnvironment
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
pc.create_index(
    name="example-index",
    dimension=1536,
    metric="cosine",
    deletion_protection='enabled',
    spec=PodSpec(
        environment=PodIndexEnvironment.EU_WEST1_GCP,
        pod_type=PodType.P1_X1
    )
)
```

#### Optional configurations for pod indexes

Pod indexes support many optional configuration fields through
the spec object. For example, the following example creates
an index that only indexes the "color" metadata field for queries
with filtering; with this metadata configuration, queries against the
index cannot filter based on any other metadata field.

This example also demonstrates horizontal scaling to multiple `replicas`.
See [Scale pod-based indexes](https://docs.pinecone.io/guides/indexes/pods/scale-pod-based-indexes) for more information on scaling.

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name="example-index-2",
    dimension=1536,
    spec=PodSpec(
        environment="eu-west1-gcp",
        pod_type='p1.x1',
        metadata_config={
            "indexed": ["color"]
        },
        replicas=2
    )
)
```
