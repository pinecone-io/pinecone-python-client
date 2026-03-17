# Backup Management Operations

Backup management operations allow you to describe and delete backups of your Pinecone indexes. These operations complement the backup creation and listing operations, enabling full lifecycle management of backups.

---

## `describe_backup()`

Retrieve detailed information about a backup.

**Source:** `pinecone/pinecone.py:1191-1210`, `pinecone/pinecone_asyncio.py:1232-1239`

| Aspect | Details |
|--------|---------|
| **Method signature** | `describe_backup(*, backup_id: str) -> BackupModel` |
| **Async signature** | `async describe_backup(*, backup_id: str) -> BackupModel` |
| **Available on** | `Pinecone`, `PineconeAsyncio` |
| **Added** | v1.0 |
| **Deprecated** | No |
| **Idempotency** | Idempotent |
| **Side effects** | None |

### Parameters

| Name | Type | Required | Default | Since | Deprecated | Description |
|------|------|----------|---------|-------|------------|-------------|
| `backup_id` | `string` | Yes | — | v1.0 | No | The unique identifier of the backup to describe. Must be a valid backup ID returned by `create_backup()` or `list_backups()`. |

### Return value

Returns a `BackupModel` object containing detailed information about the backup. See the [BackupModel data model section](#backupmodel) for complete field documentation. Key fields include:
- `backup_id`: The unique identifier of the backup
- `name`: The human-readable name of the backup
- `source_index_name`: The name of the index that was backed up
- `source_index_id`: The unique identifier of the source index
- `status`: The current status of the backup (e.g., `"Initialized"`, `"Ready"`)
- `created_at`: ISO 8601 timestamp of when the backup was created
- `record_count`: The number of vectors stored in the backup
- `namespace_count`: The number of namespaces in the backup
- `dimension`: The vector dimension of the backup
- `metric`: The distance metric used by the backup
- `description`: Optional description provided during backup creation
- `size_bytes`: The size of the backup in bytes

### Raises / Throws

| Exception / Error | Condition |
|-------------------|-----------|
| `NotFoundException` | The `backup_id` does not reference an existing backup. |
| `UnauthorizedException` | The API key is missing, invalid, or has expired. |
| `ForbiddenException` | The API key does not have permission to describe this backup. |
| `PineconeApiException` | An error occurs communicating with the Pinecone API (e.g., server error). |

### Behavior

- The backup must exist; returns `404 (not_found)` if `backup_id` does not reference an existing backup.
- The returned status field indicates whether the backup is still being created (`"Initialized"`), ready for restoration (`"Ready"`), or in another transient state.
- The `status` field is useful for polling: call this method repeatedly to check when a backup transitions from `"Initialized"` to `"Ready"` before attempting to restore from it.
- This method is read-only and does not modify the backup.

### Example

```python
from pinecone import Pinecone

pc = Pinecone()

# Describe a backup
backup = pc.describe_backup(backup_id="550e8400-e29b-41d4-a716-446655440000")

print(f"Backup name: {backup.name}")
print(f"Status: {backup.status}")
print(f"Source index: {backup.source_index_name}")
print(f"Record count: {backup.record_count}")

# Poll until backup is ready
import time
while backup.status != "Ready":
    time.sleep(5)
    backup = pc.describe_backup(backup_id="550e8400-e29b-41d4-a716-446655440000")
    print(f"Backup status: {backup.status}")
```

---

## `delete_backup()`

Delete a backup, freeing its storage.

**Source:** `pinecone/pinecone.py:1212-1228`, `pinecone/pinecone_asyncio.py:1241-1248`

| Aspect | Details |
|--------|---------|
| **Method signature** | `delete_backup(*, backup_id: str) -> None` |
| **Async signature** | `async delete_backup(*, backup_id: str) -> None` |
| **Available on** | `Pinecone`, `PineconeAsyncio` |
| **Added** | v1.0 |
| **Deprecated** | No |
| **Idempotency** | Non-idempotent |
| **Side effects** | Deletes the backup; the backup is no longer retrievable via `describe_backup()` or `list_backups()`. |

### Parameters

| Name | Type | Required | Default | Since | Deprecated | Description |
|------|------|----------|---------|-------|------------|-------------|
| `backup_id` | `string` | Yes | — | v1.0 | No | The unique identifier of the backup to delete. Must be a valid backup ID. |

### Return value

Returns `None`. The method completes successfully if the deletion was accepted. The backup is no longer accessible after this call.

### Raises / Throws

| Exception / Error | Condition |
|-------------------|-----------|
| `NotFoundException` | The `backup_id` does not reference an existing backup. |
| `UnauthorizedException` | The API key is missing, invalid, or has expired. |
| `ForbiddenException` | The API key does not have permission to delete this backup. |
| `PineconeApiException` | An error occurs communicating with the Pinecone API (e.g., server error). |

### Behavior

- The backup must exist; returns `404 (not_found)` if `backup_id` does not reference an existing backup.
- Deletion is permanent. Once a backup is deleted, it cannot be restored.
- Deleting a backup does not affect the source index — the index remains fully operational.
- Calling `describe_backup()` with the same `backup_id` after deletion will return `404 (not_found)`.
- Calling `list_backups()` after deletion will no longer include the deleted backup in the results.

### Example

```python
from pinecone import Pinecone

pc = Pinecone()

# Delete a backup
pc.delete_backup(backup_id="550e8400-e29b-41d4-a716-446655440000")

print("Backup deleted successfully")

# Attempting to describe the deleted backup will raise an error
try:
    pc.describe_backup(backup_id="550e8400-e29b-41d4-a716-446655440000")
except Exception as e:
    print(f"Error: Backup no longer exists - {type(e).__name__}")
```

---

## Data Models

### BackupModel

Represents a Pinecone backup with its configuration, status, and metadata.

**Source:** `pinecone/db_control/models/backup_model.py:12-53`, `pinecone/core/openapi/db_control/model/backup_model.py:50-144`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `backup_id` | `string (uuid)` | No | v1.0 | No | The unique identifier of the backup. Assigned by the system. |
| `name` | `string` | No | v1.0 | No | The human-readable name of the backup. Set during `create_backup()`. |
| `source_index_name` | `string` | No | v1.0 | No | The name of the index that was backed up. |
| `source_index_id` | `string (uuid)` | No | v1.0 | No | The unique identifier of the source index. |
| `status` | `string` | No | v1.0 | No | The current status of the backup. Possible values: `"Initialized"` (backup is being created), `"Ready"` (backup is ready for restoration). |
| `description` | `string` | No | v1.0 | No | Optional description provided during backup creation. Empty string if not provided. |
| `created_at` | `string (date-time)` | No | v1.0 | No | ISO 8601 timestamp of when the backup was created. |
| `dimension` | `integer (int32, 1–20000)` | No | v1.0 | No | The vector dimension of the backup. Matches the source index. |
| `metric` | `string` | No | v1.0 | No | The distance metric used by the backup (e.g., `"cosine"`, `"euclidean"`, `"dotproduct"`). |
| `record_count` | `integer (int64)` | No | v1.0 | No | The total number of vectors stored in the backup. |
| `namespace_count` | `integer (int32)` | No | v1.0 | No | The number of namespaces in the backup. |
| `size_bytes` | `integer (int64)` | No | v1.0 | No | The size of the backup in bytes. |
| `cloud` | `string` | No | v1.0 | No | The cloud provider of the backup (e.g., `"aws"`, `"gcp"`, `"azure"`). |
| `region` | `string` | No | v1.0 | No | The region where the backup is stored (e.g., `"us-east-1"`, `"eu-west-1"`). |
| `tags` | `object` | Yes | v1.0 | No | Optional tags associated with the backup, carried over from the source index. |
| `schema` | `MetadataSchema` | Yes | v1.0 | No | Optional metadata schema configuration from the source index, defining which metadata fields are indexed and filterable. |
