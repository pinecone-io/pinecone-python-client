# Backup Operations

Backup operations allow you to create and list backups of your Pinecone indexes. Backups capture the complete state of an index and can be used to restore data in new indexes or recover from accidental deletion.

---

## `Pinecone.create_backup()`

Creates a backup of a specified index.

**Source:** `pinecone/pinecone.py:1119-1150`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Not idempotent — repeated calls will fail if a backup with the same name already exists
**Side effects:** Creates a new backup resource in the Pinecone API

### Signature

```python
def create_backup(
    self,
    *,
    index_name: str,
    backup_name: str,
    description: str = ""
) -> BackupModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `index_name` | `string` | Yes | — | v1.0 | No | The name of the index to back up. Must be an existing index. |
| `backup_name` | `string` | Yes | — | v1.0 | No | A human-readable name for the backup. Must be unique within the project. |
| `description` | `string` | No | `""` | v1.0 | No | Optional description of the backup. Defaults to an empty string. |

### Returns

**Type:** `BackupModel` — An object describing the created backup. See the [BackupModel data model section](#backupmodel) for complete field documentation. Key fields include `backup_id`, `status`, `source_index_name`, `created_at`, `record_count`, and others.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. May occur if the `index_name` is invalid, if the index does not exist, if a backup with the same `backup_name` already exists, or if the API call fails. Returns `400 (validation_error)` if `index_name` does not reference an existing index. Returns `409 (conflict)` if `backup_name` is not unique within the project. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `ForbiddenException` | The API key does not have permission to create backups. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments (all arguments except `self` must be keyword-only). |

### Behavior

- The backup process initiates immediately upon calling this method. The returned `BackupModel` initially has a status of `"Initializing"` and transitions to `"Ready"` when ready for restoration.
- The backup is stored on Pinecone's infrastructure and is retained according to your account's retention policy.
- Creating a backup does not block index operations — the source index remains fully operational during backup creation.
- The backup captures the index state at the time the API call is made.

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Create a backup of an index
backup = pc.create_backup(
    index_name="my_index",
    backup_name="daily_backup_2025_03_17",
    description="Daily backup of production index"
)

print(f"Backup created with ID: {backup.backup_id}")
print(f"Status: {backup.status}")
print(f"Records: {backup.record_count}")
```

### Notes

- All arguments must be passed as keyword arguments (keyword-only).
- Backup creation is asynchronous; the `status` will initially be `"Initializing"` and will transition to `"Ready"` when the backup completes.

---

## `PineconeAsyncio.create_backup()`

Asynchronous version of `create_backup()`. Creates a backup of a specified index.

**Source:** `pinecone/pinecone_asyncio.py:1197-1209`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Not idempotent
**Side effects:** Creates a new backup resource in the Pinecone API

### Signature

```python
async def create_backup(
    self,
    *,
    index_name: str,
    backup_name: str,
    description: str = ""
) -> BackupModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `index_name` | `string` | Yes | — | v1.0 | No | The name of the index to back up. |
| `backup_name` | `string` | Yes | — | v1.0 | No | The name to assign to the backup. |
| `description` | `string` | No | `""` | v1.0 | No | Optional description for the backup. |

### Returns

**Type:** `Awaitable[BackupModel]` — An awaitable that resolves to a `BackupModel` object describing the created backup.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. May occur if the index does not exist or if a backup with the same name already exists. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `ForbiddenException` | The API key does not have permission to create backups. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def create_backup_async():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    backup = await pc.create_backup(
        index_name="my_index",
        backup_name="my_backup",
        description="Async backup"
    )
    print(f"Backup created: {backup.backup_id}")

asyncio.run(create_backup_async())
```

### Notes

- Same behavior as synchronous version, but returns an awaitable for async/await usage.

---

## `Pinecone.list_backups()`

Lists backups for a specific index or for the entire project, with pagination support.

**Source:** `pinecone/pinecone.py:1152-1189`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent — repeated calls return the same results (barring changes to backups)
**Side effects:** None — read-only operation

### Signature

```python
def list_backups(
    self,
    *,
    index_name: str | None = None,
    limit: int | None = 10,
    pagination_token: str | None = None
) -> BackupList
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `index_name` | `string` | No | `None` | v1.0 | No | The name of the index to filter backups by. When provided, only backups of this index are returned. When omitted, all backups in the project are returned. |
| `limit` | `integer (int32)` | No | `10` | v1.0 | No | The maximum number of backups to return per page. Must be between 1 and 100. Defaults to `10` when omitted. |
| `pagination_token` | `string` | No | `None` | v1.0 | No | The pagination token from a previous response to fetch the next page of results. When omitted, returns the first page. |

### Returns

**Type:** `BackupList` — An object containing: `data` (list of `BackupModel` objects), and `pagination` (object with `next` token for the next page, or `None` if no more pages).

The `BackupList` object supports:
- Direct iteration: `for backup in backups: ...`
- Indexing: `backups[0]` returns the first backup
- Length: `len(backups)` returns the number of backups in the current page
- The `names()` method: `backups.names()` returns a list of backup names

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. May occur if the `index_name` is invalid or if pagination parameters are incorrect. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

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

pc = Pinecone(api_key="sk-example-key-do-not-use")

# List all backups for a specific index
index_backups = pc.list_backups(index_name="my_index", limit=20)
print(f"Total backups: {len(index_backups.data)}")
for backup in index_backups.data:
    print(f"  - {backup.name} (ID: {backup.backup_id}, Status: {backup.status})")

# List backups for a specific index
index_backups = pc.list_backups(index_name="my_index", limit=10)
for backup in index_backups.data:
    print(f"Backup: {backup.name}, Status: {backup.status}")

# Pagination example
next_token = index_backups.pagination.next if index_backups.pagination else None
if next_token:
    page2 = pc.list_backups(limit=20, pagination_token=next_token)
    print(f"Page 2 has {len(page2.data)} backups")

# List all backups across all indexes in the project
all_backups = pc.list_backups(limit=50)
print(f"Total backups in project: {len(all_backups.data)}")
```

### Notes

- All arguments must be passed as keyword arguments.
- Default `limit` is 10. Increase to retrieve more backups per call.
- Pagination is supported via the `pagination_token` returned in the `pagination` field of the response.
- If `index_name` is not provided, the list includes all backups across all indexes in the project.

---

## `PineconeAsyncio.list_backups()`

Asynchronous version of `list_backups()`. Lists backups in the project, optionally filtered by index name.

**Source:** `pinecone/pinecone_asyncio.py:1211-1230`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

### Signature

```python
async def list_backups(
    self,
    *,
    index_name: str | None = None,
    limit: int | None = 10,
    pagination_token: str | None = None
) -> BackupList
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `index_name` | `string` | No | `None` | v1.0 | No | The name of the index to filter by. |
| `limit` | `integer (int32)` | No | `10` | v1.0 | No | The maximum number of backups to return. |
| `pagination_token` | `string` | No | `None` | v1.0 | No | Token for pagination. |

### Returns

**Type:** `Awaitable[BackupList]` — An awaitable that resolves to a `BackupList` object.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def list_backups_async():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    backups = await pc.list_backups(index_name="my_index", limit=10)
    for backup in backups.data:
        print(f"{backup.name}: {backup.status}")

asyncio.run(list_backups_async())
```

### Notes

- Same behavior as synchronous version.

---

## Common Patterns

### Polling for backup completion

```python
import time
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

backup = pc.create_backup(
    index_name="my_index",
    backup_name="my_backup"
)
backup_id = backup.backup_id

# Poll until ready
while True:
    backup = pc.describe_backup(backup_id=backup_id)
    if backup.status == "Ready":
        print(f"Backup ready with {backup.record_count} records")
        break
    elif backup.status == "Failed":
        raise Exception("Backup creation failed")
    else:
        print(f"Status: {backup.status}, waiting...")
        time.sleep(5)
```

### Iterating through all backups with pagination

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

all_backups = []
token = None

while True:
    result = pc.list_backups(limit=100, pagination_token=token)
    all_backups.extend(result.data)

    # Check if there are more pages
    if not result.pagination or not result.pagination.next:
        break

    token = result.pagination.next

print(f"Total backups: {len(all_backups)}")
```

---

## Data Models

### `BackupModel`

Represents a Pinecone backup with its configuration, status, and metadata.

**Source:** `pinecone/db_control/models/backup_model.py:12-53`, `pinecone/core/openapi/db_control/model/backup_model.py:50-144`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `backup_id` | `string (uuid)` | No | v1.0 | No | The unique identifier of the backup. Assigned by the system. |
| `name` | `string` | Yes | v1.0 | No | The human-readable name of the backup. Set during `create_backup()`. |
| `source_index_name` | `string` | No | v1.0 | No | The name of the index that was backed up. |
| `source_index_id` | `string (uuid)` | No | v1.0 | No | The unique identifier of the source index. |
| `status` | `string` | No | v1.0 | No | The current status of the backup. Possible values: `"Initializing"` (backup is being created), `"Ready"` (backup is ready for restoration), `"Failed"`. |
| `description` | `string` | Yes | v1.0 | No | Optional description provided during backup creation. Empty string if not provided. |
| `created_at` | `string (date-time)` | Yes | v1.0 | No | ISO 8601 timestamp of when the backup was created. |
| `dimension` | `integer (int32, 1–20000)` | Yes | v1.0 | No | The vector dimension of the backup. Matches the source index. |
| `metric` | `string` | Yes | v1.0 | No | The distance metric used by the backup (e.g., `"cosine"`, `"euclidean"`, `"dotproduct"`). |
| `record_count` | `integer (int64)` | Yes | v1.0 | No | The total number of records in the backed up index. `null` or zero until status is `"Ready"`. |
| `namespace_count` | `integer (int32)` | Yes | v1.0 | No | The number of namespaces in the backed up index. |
| `size_bytes` | `integer (int64)` | Yes | v1.0 | No | The size of the backup in bytes. `null` or zero until status is `"Ready"`. |
| `cloud` | `string` | No | v1.0 | No | The cloud provider of the backup (e.g., `"aws"`, `"gcp"`, `"azure"`). |
| `region` | `string` | No | v1.0 | No | The region where the backup is stored (e.g., `"us-east-1"`, `"eu-west-1"`). |
| `tags` | `object` | Yes | v1.0 | No | Optional tags associated with the backup, carried over from the source index. |
| `schema` | `MetadataSchema` | Yes | v1.0 | No | Optional metadata schema configuration from the source index, defining which metadata fields are indexed and filterable. |

### `BackupList`

A container for a paginated list of backups.

**Source:** `pinecone/db_control/models/backup_list.py:6-48`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `data` | `array of BackupModel` | No | v1.0 | No | The backups in the current page. |
| `pagination` | `object (PaginationResponse)` | Yes | v1.0 | No | Pagination metadata including `next` token for fetching the subsequent page. |

The `PaginationResponse` object has:

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `next` | `string` | Yes | The pagination token to fetch the next page of results. `null` or omitted if there are no more pages. |
