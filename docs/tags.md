# Labeling indexes with index tags

Tags are key-value pairs you can attach to indexes to better understand, organize, and identify your resources. Tags are flexible and can be tailored to your needs, but some common use cases for them might be to label an index with the relevant deployment `environment`, `application`, `team`, or `owner`.

Tags can be set during index creation by passing an optional dictionary with the `tags` keyword argument.

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
    ),
    tags={
        "environment": "testing",
        "owner": "jsmith",
    }
)
```

## Modifying tags

To modify the tags of an existing index, use `configure_index()`. The `configure_index` method can be used to change the value of an existing tag, add new tags, or delete tags by setting the value to empty string.

Here's an example showing different ways of modifying tags.

## Viewing tags with describe_index

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

desc = pc.describe_index(name='my-index')
print(desc.tags)
# {
#   'environment': 'testing',
#   'owner': 'jsmith'
# }
```

## Modifying an existing tag with configure_index

You can modify an existing tag by passing the key-value pair to `configure_index()`. Other tags on the index will not be changed.

````python
pc.configure_index(
    name='my-index',
    tags={"environment": "production"}
)

desc = pc.describe_index(
    name=index_name,
)
print(desc.tags)
# {
#   'environment': 'production',
#   'owner': 'jsmith'
# }
````

## Adding a new tag to an existing index

Tags passed to the `configure_index` method are merged with any tags that an index already has.

```python
pc.configure_index(
    name='my-index',
    tags={"purpose": "for testing the new chatbot feature"}
)

desc = pc.describe_index(
    name=index_name,
)
print(desc.tags)
# {
#     'purpose': 'for testing the new chatbot feature',
#     'environment': 'production',
#     'owner': 'jsmith'
# }
```

## Removing a tag

To remove a tag, pass the value empty string.

```python
pc.configure_index(
    name='my-index',
    tags={
        "purpose": "",
        "environment": "",
        "owner": ""
    }
)
```
