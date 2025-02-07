# Pinecone serverless indexes

For introductory information on indexes, please see [Understanding indexes](https://docs.pinecone.io/guides/indexes/understanding-indexes#serverless-indexes)

## Create a serverless index on Amazon Web Services (AWS)

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

## Create a serverless index on Google Cloud Platform

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

## Create a serverless index on Azure

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
