# Managing indexes

This page describes operations which are available for all index types.

## List indexes

The following example returns all indexes in your project.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')
for index in pc.list_indexes():
    print(index['name'])
```

## Describe index

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

pc.delete_index("example-index")
```

## Configure index

All indexes can have their configurations modified using the `configure_index` method, although not all indexes will support all properties.
`configure_index` can be used to modify [tags](shared-index-configs.md#tags) and [deletion_protection][shared-index-configs.md#deletion-protection]
