
# Collections

## Create collection

The following example creates the collection `example-collection` from a pod index named `example-index`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.create_collection(
    name="example-collection",
    source="example-index"
)
```

## List collections

The following example returns a list of the collections in the current project.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

active_collections = pc.list_collections()
```

## Describe a collection

The following example returns a description of the collection
`example-collection`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

collection_description = pc.describe_collection("example-collection")
```

## Delete a collection

The following example deletes the collection `example-collection`.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

pc.delete_collection("example-collection")
```
