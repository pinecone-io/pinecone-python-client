# Working with collections

A collection is a read-only, static snapshot of a **pod-based** index's vector data, held
outside the index. Collections and backups are separate mechanisms for different deployment
types, and they are not interchangeable:

| | Snapshot mechanism | Taken with | Restored with |
|---|---|---|---|
| Pod-based index | Collection | `pc.collections.create()` | a new index created from it, on API versions before `2026-07` only; see [below](#create-an-index-from-a-collection) |
| Managed (serverless) or BYOC index | Backup | `pc.backups.create()` | `pc.create_index_from_backup()` |

For backups, see [backups and restore](indexes/backups-and-restore.md).

```{important}
API version `2026-07`, the version this SDK targets, does not create pod-based indexes, so
there is no source for `pc.collections.create()` to copy. `list()`, `describe()`, and
`delete()` still work on collections that already exist, on both `Pinecone` and
`AsyncPinecone`. See [pod-based indexes](indexes/pod.md), and {ref}`pod-collections` in the
v10 migration guide for the API-version pin that reaches an earlier version.
```

## Create a collection

{meth}`~pinecone.client.collections.Collections.create` takes a name for the snapshot and
the name of the pod-based index to copy:

```python
col = pc.collections.create(name="movie-embeddings-snapshot", source="movie-recommendations")
print(col.status)   # "Initializing"
```

The source must be a pod-based index that is ready; the server rejects a serverless or BYOC
source. The call returns as soon as creation starts and takes no `timeout=`, so poll
`describe` until the status leaves `"Initializing"`.

## List collections

{meth}`~pinecone.client.collections.Collections.list` returns every collection in the
project at once, with no pagination, as a
{class}`~pinecone.models.collections.list.CollectionList`:

```python
for col in pc.collections.list():
    print(col.name, col.status)

print(pc.collections.list().names())   # ['movie-embeddings-snapshot', 'product-catalog-snapshot']
```

Use it to find leftover collections before deleting a project: the server refuses to delete
a project that still holds collections, with
{exc}`~pinecone.errors.exceptions.FailedPreconditionError`.

## Describe a collection

{meth}`~pinecone.client.collections.Collections.describe` returns a
{class}`~pinecone.models.collections.model.CollectionModel`:

```python
col = pc.collections.describe("movie-embeddings-snapshot")
print(col.status)        # 'Ready'
print(col.dimension)     # 1024
print(col.vector_count)  # 99
print(col.size)          # 3126700
```

`size` is the space the snapshot occupies in bytes. `size`, `dimension`, and
`vector_count` are `None` until the collection finishes initializing, so check `status`
first.

## Delete a collection

```python
pc.collections.delete("movie-embeddings-snapshot")
```

Deletion is asynchronous: the call returns once the request is accepted, and the collection
can appear in `list` for a short time afterwards. While a collection is still being created
or deleted, the server refuses to delete its source index with
{exc}`~pinecone.errors.exceptions.FailedPreconditionError`.

## Create an index from a collection

API versions before `2026-07` restore a collection by creating a new pod-based index from
it with `source_collection`; see Pinecone's
[Restore a pod-based index](https://docs.pinecone.io/guides/indexes/pods/restore-a-pod-based-index).
`2026-07` rejects index creation from a collection with a 400.

This SDK cannot make that request on any API version. It raises
{exc}`~pinecone.errors.exceptions.PineconeTypeError` before sending, whether
`source_collection` is passed to {meth}`~pinecone.client.indexes.Indexes.create` directly or
set on a {class}`~pinecone.models.indexes.specs.PodSpec` passed to the deprecated `spec=`
argument, and an `X-Pinecone-Api-Version` pin does not change that. To restore a collection,
use a client that targets an earlier API version, or call the REST API with an earlier
version header. Backups of serverless and BYOC indexes restore with
{meth}`~pinecone.Pinecone.create_index_from_backup`; see
[backups and restore](indexes/backups-and-restore.md).
