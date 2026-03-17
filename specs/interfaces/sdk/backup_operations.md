# Backup Operations

Backup operations allow you to create, list, and manage backups of your Pinecone indexes. Backups capture the complete state of an index and can be used to restore data in new indexes or recover from accidental deletion.

---

## `create_backup()`

Creates a backup of a specified index.

**Source:** `pinecone/pinecone.py:1120-1150`

| Aspect | Details |
|--------|---------|
| **Method signature** | `create_backup(*, index_name: str, backup_name: str, description: str = "") -> BackupModel` |
| **Available on** | `Pinecone`, `PineconeAsyncio` |

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `index_name` | `string` | Yes | The name of the index to back up. Must be an existing index. |
| `backup_name` | `string` | Yes | A human-readable name for the backup. Must be unique within the project. |
| `description` | `string` | No | Optional description of the backup. Defaults to an empty string. |

### Return value

Returns a `BackupModel` object representing the created backup. See the [BackupModel data model section](#backupmodel) for complete field documentation. Key fields include `backup_id`, `status`, `source_index_name`, `created_at`, `record_count`, and others.

### Behavior

- The backup process initiates immediately upon calling this method. The returned `BackupModel` initially has a status of `"initialized"` and transitions to `"completed"` when ready for restoration.
- The backup is stored on Pinecone's infrastructure and is retained according to your account's retention policy.
- Creating a backup does not block index operations — the source index remains fully operational during backup creation.
- Returns `400 (validation_error)` if `index_name` does not reference an existing index.
- Returns `409 (conflict)` if `backup_name` is not unique within the project.

### Example

```python
from pinecone import Pinecone

pc = Pinecone()

# Create a backup of an index
backup = pc.create_backup(
    index_name="my_index",
    backup_name="daily_backup_2025_03_17",
    description="Daily backup of production index"
)

print(f"Backup created with ID: {backup.backup_id}")
print(f"Status: {backup.status}")
```

---

## `list_backups()`

Lists backups for a specific index or for the entire project, with pagination support.

**Source:** `pinecone/pinecone.py:1152-1189`

| Aspect | Details |
|--------|---------|
| **Method signature** | `list_backups(*, index_name: str \| None = None, limit: int \| None = 10, pagination_token: str \| None = None) -> BackupList` |
| **Available on** | `Pinecone`, `PineconeAsyncio` |

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `index_name` | `string` | No | The name of the index to filter backups by. When provided, only backups of this index are returned. When omitted, all backups in the project are returned. |
| `limit` | `integer (int32)` | No | The maximum number of backups to return per page. Must be between 1 and 100. Defaults to `10` when omitted. |
| `pagination_token` | `string` | No | The pagination token from a previous response to fetch the next page of results. When omitted, returns the first page. |

### Return value

Returns a `BackupList` object, which is an iterable container of `BackupModel` objects with pagination metadata.

| Property | Type | Description |
|----------|------|-------------|
| `data` | `array of BackupModel` | The list of backups in the current page. |
| `pagination` | `object` | Pagination metadata for subsequent requests. Contains `next` field with the pagination token to fetch the next page. |

The `BackupList` object supports:
- Direct iteration: `for backup in backups: ...`
- Indexing: `backups[0]` returns the first backup
- Length: `len(backups)` returns the number of backups in the current page
- The `names()` method: `backups.names()` returns a list of backup names

### Behavior

- When `index_name` is omitted, returns all backups across all indexes in the project.
- Results are ordered by creation time, most recent first.
- The `pagination` field is present only if there are additional pages to fetch.
- When the `pagination` field is absent or `pagination.next` is `null`, you have reached the last page.
- The `limit` parameter affects only the current page size — it does not limit the total number of items you can retrieve across all pages.
- If `limit` exceeds 100, the API clamps it to 100.
- If `limit` is 0 or negative, returns `400 (validation_error)`.
- If `index_name` is provided but the index does not exist, returns `404 (not_found)`.

### Example

```python
from pinecone import Pinecone

pc = Pinecone()

# List all backups for a specific index
index_backups = pc.list_backups(index_name="my_index", limit=20)

for backup in index_backups:
    print(f"Backup: {backup.name}, Status: {backup.status}")

# Paginate through results
pagination_token = None
while True:
    backups = pc.list_backups(
        index_name="my_index",
        limit=50,
        pagination_token=pagination_token
    )

    for backup in backups:
        print(f"{backup.name}: {backup.status}")

    # Check if there are more results
    if backups.pagination and backups.pagination.next:
        pagination_token = backups.pagination.next
    else:
        break

# List all backups across all indexes in the project
all_backups = pc.list_backups(limit=50)
print(f"Total backups in project: {len(all_backups)}")
```

---

## Data Models

### BackupModel

**Source:** `pinecone/db_control/models/backup_model.py:12-54`, `pinecone/core/openapi/db_control/model/backup_model.py:104-121`

Represents a backup of a Pinecone index.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `backup_id` | `string` | No | The server-assigned unique identifier for the backup. |
| `name` | `string` | Yes | The human-readable name of the backup. |
| `status` | `string` | No | The current status of the backup. Values: `initialized`, `completed`, `failed`. |
| `source_index_name` | `string` | No | The name of the source index. |
| `source_index_id` | `string` | No | The unique ID of the source index. |
| `cloud` | `string` | No | The cloud provider where the backup is stored (e.g., `aws`, `gcp`, `azure`). |
| `region` | `string` | No | The cloud region where the backup is stored (e.g., `us-east-1`). |
| `created_at` | `string (date-time)` | No | ISO 8601 timestamp when the backup was created. |
| `size_bytes` | `integer (int64)` | Yes | The size of the backup in bytes. `null` or zero until status is `completed`. |
| `record_count` | `integer (int64)` | Yes | The total number of records in the backed up index. `null` or zero until status is `completed`. |
| `namespace_count` | `integer (int32)` | Yes | The number of namespaces in the backed up index. |
| `dimension` | `integer (int32)` | Yes | The dimensionality of vectors in the backed up index. |
| `metric` | `string` | Yes | The distance metric configured for the backed up index (e.g., `cosine`, `euclidean`, `dotproduct`). |
| `description` | `string` | Yes | The optional description provided during backup creation. |
| `schema` | `object (MetadataSchema)` | Yes | The metadata schema configuration from the backed up index. Describes which metadata fields are indexed and filterable. |
| `tags` | `object (IndexTags)` | Yes | Tags attached to the source index. |

### BackupList

**Source:** `pinecone/db_control/models/backup_list.py:6-49`

A container for a paginated list of backups.

| Field | Type | Description |
|-------|------|-------------|
| `data` | `array of BackupModel` | The backups in the current page. |
| `pagination` | `object (PaginationResponse)` | Pagination metadata including `next` token for fetching the subsequent page. |

The `PaginationResponse` object has:

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `next` | `string` | Yes | The pagination token to fetch the next page of results. `null` or omitted if there are no more pages. |
