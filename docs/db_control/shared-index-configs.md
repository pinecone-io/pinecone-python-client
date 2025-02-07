# Tags

## Organizing indexes with index tags

Tags are key-value pairs you can attach to indexes to better understand, organize, and identify your resources. Tags are flexible and can be tailored to your needs, but some common use cases for them might be to label an index with the relevant deployment `environment`, `application`, `team`, or `owner`.

Tags can be set during index creation by passing an optional dictionary with the `tags` keyword argument to the `create_index` and `create_index_for_model` methods. Here's an example demonstrating how tags can be passed to `create_index`.

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

# Deletion Protection

Deletion protection prevents indexes from being accidentally deleted.

The `deletion_protection` configuration is available on all index types that can be set to either `"enabled"` or `"disabled"`. When enabled, an index will not be able to be deleted using the `delete_index` method. The setting can be changed after an index is created using the `configure_index` method.

## Enabling deletion protection during index creation.

Deletion protection is disabled by default, but you can enable it at the time an index is created by passing an optional keyword argument to `create_index` or `create_index_for_model`.

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
    deletion_protection=DeletionProtection.ENABLED,
    spec=ServerlessSpec(
        cloud=CloudProvider.GCP,
        region=GcpRegion.US_CENTRAL1
    )
)
```


## Configuring deletion protection

If you would like to enable deletion protection, which prevents an index from being deleted, the `configure_index` method also handles that via an optional `deletion_protection` keyword argument.

```python
from pinecone import Pinecone

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

# To enable deletion protection
pc.configure_index("example-index", deletion_protection='enabled')

# Disable deletion protection
pc.configure_index("example-index", deletion_protection='disabled')

# Call describe index to verify the configuration change has been applied
desc = pc.describe_index("example-index")
print(desc.deletion_protection)
```
