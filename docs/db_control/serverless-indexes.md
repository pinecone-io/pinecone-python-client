# Serverless Indexes

For introductory information on indexes, please see [Understanding indexes](https://docs.pinecone.io/guides/indexes/understanding-indexes#serverless-indexes)

## Sparse vs Dense embedding vectors

When you are working with dense embedding vectors, you must specify the `dimension` of the vectors you expect to store at the time your index is created. For sparse vectors, used to represent vectors where most values are zero, you omit `dimension` and must specify `vector_type="sparse"`.

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    Metric,
    VectorType
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name='index-for-dense-vectors',
    dimension=1536,
    metric=Metric.COSINE,
    # vector_type="dense" is the default value, so it can be omitted if you prefer
    vector_type=VectorType.DENSE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_WEST_2
    ),
)

pc.create_index(
    name='index-for-sparse-vectors',
    metric=Metric.DOTPRODUCT,
    vector_type=VectorType.SPARSE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_WEST_2
    ),
)
```

## Available clouds

See the [available cloud regions](https://docs.pinecone.io/troubleshooting/available-cloud-regions) page for the most up-to-date information one which cloud regions are available.

### Create a serverless index on Amazon Web Services (AWS)

The following example creates a serverless index in the `us-west-2`
region of AWS. For more information on serverless and regional availability, see [Understanding indexes](https://docs.pinecone.io/guides/indexes/understanding-indexes#serverless-indexes).

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    Metric,
    VectorType
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_WEST_2
    ),
    vector_type=VectorType.DENSE
)
```

### Create a serverless index on Google Cloud Platform

The following example creates a serverless index in the `us-central1`
region of GCP. For more information on serverless and regional availability, see [Understanding indexes](https://docs.pinecone.io/guides/indexes/understanding-indexes#serverless-indexes).

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    GcpRegion,
    Metric
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.GCP,
        region=GcpRegion.US_CENTRAL1
    )
)
```

### Create a serverless index on Azure

The following example creates a serverless index on Azure. For more information on serverless and regional availability, see [Understanding indexes](https://docs.pinecone.io/guides/indexes/understanding-indexes#serverless-indexes).

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AzureRegion,
    Metric
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AZURE,
        region=AzureRegion.EASTUS2
    )
)
```

## Read Capacity Configuration

You can configure the read capacity mode for your serverless index. By default, indexes are created with `OnDemand` mode. You can also specify `Dedicated` mode with dedicated read nodes.

### Dedicated Read Capacity

Dedicated mode allocates dedicated read nodes for your workload. You must specify `node_type`, `scaling`, and scaling configuration.

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    GcpRegion,
    Metric
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.GCP,
        region=GcpRegion.US_CENTRAL1,
        read_capacity={
            "mode": "Dedicated",
            "dedicated": {
                "node_type": "t1",
                "scaling": "Manual",
                "manual": {
                    "shards": 2,
                    "replicas": 2
                }
            }
        }
    )
)
```

### Configuring Read Capacity

You can change the read capacity configuration of an existing serverless index using `configure_index`. This allows you to:

- Switch between OnDemand and Dedicated modes
- Adjust the number of shards and replicas for Dedicated mode with manual scaling

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

# Switch to OnDemand read capacity
pc.configure_index(
    name='my-index',
    read_capacity={"mode": "OnDemand"}
)

# Switch to Dedicated read capacity with manual scaling
pc.configure_index(
    name='my-index',
    read_capacity={
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {
                "shards": 3,
                "replicas": 2
            }
        }
    }
)

# Scale up by increasing shards and replicas
pc.configure_index(
    name='my-index',
    read_capacity={
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {
                "shards": 4,
                "replicas": 3
            }
        }
    }
)
```

When you change read capacity configuration, the index will transition to the new configuration. You can use `describe_index` to check the status of the transition.

## Metadata Schema Configuration

You can configure which metadata fields are filterable by specifying a metadata schema. By default, all metadata fields are indexed. However, large amounts of metadata can cause slower index building as well as slower query execution, particularly when data is not cached in a query executor's memory and local SSD and must be fetched from object storage.

To prevent performance issues due to excessive metadata, you can limit metadata indexing to the fields that you plan to use for query filtering. When you specify a metadata schema, only fields marked as `filterable: True` are indexed and can be used in filters.

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    Metric
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_WEST_2,
        schema={
            "genre": {"filterable": True},
            "year": {"filterable": True},
            "description": {"filterable": True}
        }
    )
)
```

## Configuring, listing, describing, and deleting

See [shared index actions](shared-index-actions.md) to learn about how to manage the lifecycle of your index after it is created.
