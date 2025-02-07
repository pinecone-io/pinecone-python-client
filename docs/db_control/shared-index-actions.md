# Managing indexes

This page describes operations which are available for all index types.

## Check if an index exists

This `has_index` method can do a simple boolean check on whether an index exists.

```python
from pinecone import Pinecone, ServerlessSpec, AwsRegion

pc = Pinecone()

index_name = "my_index"
if not pc.has_index(name=index_name):
    print("Index does not exist, creating...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=AwsRegion.US_WEST_2)
    )
```

## List indexes

The following example returns all indexes in your project.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

for index in pc.list_indexes():
    print(index)
```

To see just the names, the response object has a convenience method `names()` which returns an iterator.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

for name in pc.list_indexes().names():
    print("index name is: ", name)
```

## Describe index

The `describe_index` method is used to fetch a complete description of an index's configuration. This description
includes critical information such as the `host` used to connect to the index and perform data operations.

The following example returns information about the index `example-index`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

index_description = pc.describe_index(name="book-search")
# {
#     "name": "book-search",
#     "metric": "cosine",
#     "host": "book-search-dojoi3u.svc.aped-4627-b74a.pinecone.io",
#     "spec": {
#         "serverless": {
#             "cloud": "aws",
#             "region": "us-east-1"
#         }
#     },
#     "status": {
#         "ready": true,
#         "state": "Ready"
#     },
#     "vector_type": "dense",
#     "dimension": 1024,
#     "deletion_protection": "disabled",
#     "tags": null,
# }
```

## Delete an index

The following example deletes the index named `example-index`. Only indexes which are not protected by deletion protection may be deleted.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.delete_index(name="example-index")
```

## Configure index

All indexes can have their configurations modified using the `configure_index` method, although not all indexes will support all properties.
The `configure_index` method can be used to modify [tags](shared-index-configs.md#tags) and [deletion_protection][shared-index-configs.md#deletion-protection].
For pod-based indexes, options are accepted to help with scaling. See [Scaling pod-based indexes](https://docs.pinecone.io/guides/indexes/pods/scale-pod-based-indexes) for more info.
