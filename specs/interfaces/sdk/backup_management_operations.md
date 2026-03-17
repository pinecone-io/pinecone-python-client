# Backup Management Operations

Backup management operations allow you to describe and delete backups of your Pinecone indexes. These operations complement the backup creation and listing operations in [backup_operations.md](backup_operations.md), enabling full lifecycle management of backups.

---

## `Pinecone.describe_backup()`

Retrieves detailed information about a specific backup.

**Source:** `pinecone/pinecone.py:1191-1210`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

### Signature

```python
def describe_backup(self, *, backup_id: str) -> BackupModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `backup_id` | `string` | Yes | — | v1.0 | No | The unique identifier of the backup to describe. Must be a valid backup ID returned by `create_backup()` or `list_backups()`. |

### Returns

**Type:** `BackupModel` — An object containing detailed information about the backup. See [backup_operations.md](backup_operations.md#backupmodel) for complete field documentation. Key fields include:
- `backup_id`: The unique identifier of the backup
- `name`: The human-readable name of the backup
- `source_index_name`: The name of the index that was backed up
- `source_index_id`: The unique identifier of the source index
- `status`: The current status of the backup (e.g., `"Initialized"`, `"Ready"`, `"Failed"`)
- `created_at`: ISO 8601 timestamp of when the backup was created
- `record_count`: The number of vectors stored in the backup
- `namespace_count`: The number of namespaces in the backup
- `dimension`: The vector dimension of the backup
- `metric`: The distance metric used by the backup
- `description`: Optional description provided during backup creation
- `size_bytes`: The size of the backup in bytes

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The `backup_id` does not reference an existing backup. Returns `404 (not_found)`. |
| `UnauthorizedException` | The API key is missing, invalid, or has expired. |
| `ForbiddenException` | The API key does not have permission to describe this backup. |
| `PineconeApiException` | An error occurs communicating with the Pinecone API (e.g., server error). |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

### Behavior

- The backup must exist; returns `404 (not_found)` if `backup_id` does not reference an existing backup.
- The returned status field indicates whether the backup is still being created (`"Initialized"`), ready for restoration (`"Ready"`), or has failed (`"Failed"`).
- The `status` field is useful for polling: call this method repeatedly to check when a backup transitions from `"Initialized"` to `"Ready"` before attempting to restore from it.
- This method is read-only and does not modify the backup.
- Status transitions: `"Initialized"` -> `"Ready"` or `"Failed"`.

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Describe a backup
backup = pc.describe_backup(backup_id="550e8400-e29b-41d4-a716-446655440000")

print(f"Backup name: {backup.name}")
print(f"Status: {backup.status}")
print(f"Source index: {backup.source_index_name}")
print(f"Record count: {backup.record_count}")
print(f"Created: {backup.created_at}")
print(f"Size: {backup.size_bytes} bytes")
print(f"Cloud: {backup.cloud}, Region: {backup.region}")

# Poll until backup is ready
import time
while backup.status != "Ready":
    time.sleep(5)
    backup = pc.describe_backup(backup_id="550e8400-e29b-41d4-a716-446655440000")
    print(f"Backup status: {backup.status}")
```

### Notes

- All arguments must be passed as keyword arguments.
- Use this method to check backup status during the backup creation process.

---

## `PineconeAsyncio.describe_backup()`

Asynchronous version of `describe_backup()`. Retrieves detailed information about a specific backup.

**Source:** `pinecone/pinecone_asyncio.py:1232-1239`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

### Signature

```python
async def describe_backup(self, *, backup_id: str) -> BackupModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `backup_id` | `string` | Yes | — | v1.0 | No | The ID of the backup to describe. |

### Returns

**Type:** `Awaitable[BackupModel]` — An awaitable that resolves to a `BackupModel` object.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The backup does not exist. |
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def describe_backup_async():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    backup = await pc.describe_backup(backup_id="backup-123")
    print(f"{backup.name}: {backup.status}")

asyncio.run(describe_backup_async())
```

### Notes

- Same behavior as synchronous version, but returns an awaitable for async/await usage.

---

## `Pinecone.delete_backup()`

Deletes a backup permanently, freeing its storage.

**Source:** `pinecone/pinecone.py:1212-1228`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent — repeated calls do not raise an error if the backup is already deleted
**Side effects:** Permanently deletes the backup resource and all associated data

### Signature

```python
def delete_backup(self, *, backup_id: str) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `backup_id` | `string` | Yes | — | v1.0 | No | The unique identifier of the backup to delete. Must be a valid backup ID. |

### Returns

**Type:** `None` — This method returns nothing on success. The backup is no longer accessible after this call.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The `backup_id` does not reference an existing backup. Returns `404 (not_found)`. May only occur on first call; subsequent calls are idempotent. |
| `UnauthorizedException` | The API key is missing, invalid, or has expired. |
| `ForbiddenException` | The API key does not have permission to delete this backup. |
| `PineconeApiException` | An error occurs communicating with the Pinecone API (e.g., server error). |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

### Behavior

- The backup must exist; returns `404 (not_found)` if `backup_id` does not reference an existing backup.
- Deletion is permanent. Once a backup is deleted, it cannot be restored.
- Deleting a backup does not affect the source index — the index remains fully operational.
- Calling `describe_backup()` with the same `backup_id` after deletion will return `404 (not_found)`.
- Calling `list_backups()` after deletion will no longer include the deleted backup in the results.
- Repeated calls to delete the same backup are safe (idempotent).

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Delete a backup
pc.delete_backup(backup_id="550e8400-e29b-41d4-a716-446655440000")

print("Backup deleted successfully")

# Attempting to describe the deleted backup will raise an error
try:
    pc.describe_backup(backup_id="550e8400-e29b-41d4-a716-446655440000")
except Exception as e:
    print(f"Error: Backup no longer exists - {type(e).__name__}")
```

### Notes

- All arguments must be passed as keyword arguments.
- This operation is permanent and cannot be undone.

---

## `PineconeAsyncio.delete_backup()`

Asynchronous version of `delete_backup()`. Deletes a backup permanently.

**Source:** `pinecone/pinecone_asyncio.py:1241-1248`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** Permanently deletes the backup resource

### Signature

```python
async def delete_backup(self, *, backup_id: str) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `backup_id` | `string` | Yes | — | v1.0 | No | The ID of the backup to delete. |

### Returns

**Type:** `Awaitable[None]` — An awaitable that resolves to `None` on success.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The backup does not exist. |
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def delete_backup_async():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")
    await pc.delete_backup(backup_id="backup-123")
    print("Backup deleted")

asyncio.run(delete_backup_async())
```

### Notes

- Same behavior as synchronous version, but returns an awaitable for async/await usage.

---

## Data Models

### `BackupModel`

See [backup_operations.md](backup_operations.md#backupmodel) for the complete `BackupModel` data model definition.
