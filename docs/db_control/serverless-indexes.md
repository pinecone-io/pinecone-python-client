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

# Configuring, listing, describing, and deleting

See [shared index actions](shared-index-actions.md) to learn about how to manage the lifecycle of your index after it is created.
