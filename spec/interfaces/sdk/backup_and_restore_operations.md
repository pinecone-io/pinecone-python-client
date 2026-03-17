# Backup and Restore Operations

This module documents backup and restore management operations on the Pinecone and PineconeAsyncio clients: creating backups, listing and describing backups, deleting backups, and monitoring restore jobs. Backups provide point-in-time snapshots of your indexes for disaster recovery and data protection.

## Overview

**Language / runtime:** Python 3.8+
**Package:** `pinecone`
**Module:** `pinecone`
**Class:** `Pinecone` and `PineconeAsyncio`
**Version:** v8.1.0
**Breaking change definition:** Changing the return type or return value structure of any method, removing a method, or renaming a parameter.

## Methods

### `Pinecone.create_backup(*, index_name: str, backup_name: str, description: str = "") -> BackupModel`

Creates a backup of an index.

**Import:** `from pinecone import Pinecone, BackupModel`
**Source:** `pinecone/pinecone.py:1119-1150`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Not idempotent — repeated calls will fail if a backup with the same name already exists
**Side effects:** Creates a new backup resource in the Pinecone API

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| index_name | str | Yes | — | The name of the index to backup. The index must exist. |
| backup_name | str | Yes | — | The name to assign to the backup. Must be unique within the project. |
| description | str | No | `""` | An optional description providing context for the backup. |

**Returns:** `BackupModel` — An object describing the created backup with fields: `backup_id`, `source_index_name`, `source_index_id`, `status`, `cloud`, `region`, `name`, `description`, `dimension`, `metric`, `record_count`, `namespace_count`, `size_bytes`, `created_at`.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The request failed. May occur if the `index_name` is invalid, if the index does not exist, if a backup with the same `backup_name` already exists, or if the API call fails. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments (all arguments except `self` must be keyword-only). |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Create a backup of an index
backup = pc.create_backup(
    index_name="my_index",
    backup_name="my_backup",
    description="Daily backup"
)

print(f"Backup created with ID: {backup.backup_id}")
print(f"Status: {backup.status}")
print(f"Records: {backup.record_count}")
```

**Notes**

- All arguments must be passed as keyword arguments (keyword-only).
- Backup creation is asynchronous; the `status` will initially be "Initializing" and will transition to "Ready" when the backup completes.
- The backup captures the index state at the time the API call is made.

---

### `PineconeAsyncio.create_backup(*, index_name: str, backup_name: str, description: str = "") -> Awaitable[BackupModel]`

Asynchronous version of `create_backup()`. Creates a backup of an index.

**Import:** `from pinecone import PineconeAsyncio, BackupModel`
**Source:** `pinecone/pinecone_asyncio.py:1197-1209`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Not idempotent
**Side effects:** Creates a new backup resource in the Pinecone API

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| index_name | str | Yes | — | The name of the index to backup. |
| backup_name | str | Yes | — | The name to assign to the backup. |
| description | str | No | `""` | Optional description for the backup. |

**Returns:** `Awaitable[BackupModel]` — An awaitable that resolves to a `BackupModel` object describing the created backup.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The request failed. May occur if the index does not exist or if a backup with the same name already exists. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

**Example**

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

**Notes**

- Same behavior as synchronous version, but returns an awaitable for async/await usage.

---

### `Pinecone.list_backups(*, index_name: str | None = None, limit: int | None = 10, pagination_token: str | None = None) -> BackupList`

Lists backups in the project, optionally filtered by index name.

**Import:** `from pinecone import Pinecone, BackupList`
**Source:** `pinecone/pinecone.py:1152-1189`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent — repeated calls return the same results (barring changes to backups)
**Side effects:** None — read-only operation

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| index_name | str \| None | No | `None` | The name of the index to filter by. If `None`, all backups in the project are returned. |
| limit | int \| None | No | `10` | The maximum number of backups to return per page. |
| pagination_token | str \| None | No | `None` | Token for pagination to retrieve subsequent pages of results. |

**Returns:** `BackupList` — An object containing: `data` (list of `BackupModel` objects), and `pagination` (object with `next` token for the next page, or `None` if no more pages).

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The request failed. May occur if the `index_name` is invalid or if pagination parameters are incorrect. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# List all backups in the project
all_backups = pc.list_backups(limit=20)
print(f"Total backups: {len(all_backups.data)}")
for backup in all_backups.data:
    print(f"  - {backup.name} (ID: {backup.backup_id}, Status: {backup.status})")

# List backups for a specific index
index_backups = pc.list_backups(index_name="my_index", limit=10)
for backup in index_backups.data:
    print(f"Backup: {backup.name}, Status: {backup.status}")

# Pagination example
next_token = all_backups.pagination.next if all_backups.pagination else None
if next_token:
    page2 = pc.list_backups(limit=20, pagination_token=next_token)
    print(f"Page 2 has {len(page2.data)} backups")
```

**Notes**

- All arguments must be passed as keyword arguments.
- Default `limit` is 10. Increase to retrieve more backups per call.
- Pagination is supported via the `pagination_token` returned in the `pagination` field of the response.
- If `index_name` is not provided, the list includes all backups across all indexes in the project.

---

### `PineconeAsyncio.list_backups(*, index_name: str | None = None, limit: int | None = 10, pagination_token: str | None = None) -> Awaitable[BackupList]`

Asynchronous version of `list_backups()`. Lists backups in the project, optionally filtered by index name.

**Import:** `from pinecone import PineconeAsyncio, BackupList`
**Source:** `pinecone/pinecone_asyncio.py:1211-1230`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| index_name | str \| None | No | `None` | The name of the index to filter by. |
| limit | int \| None | No | `10` | The maximum number of backups to return. |
| pagination_token | str \| None | No | `None` | Token for pagination. |

**Returns:** `Awaitable[BackupList]` — An awaitable that resolves to a `BackupList` object.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

**Example**

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

**Notes**

- Same behavior as synchronous version.

---

### `Pinecone.describe_backup(*, backup_id: str) -> BackupModel`

Retrieves detailed information about a specific backup.

**Import:** `from pinecone import Pinecone, BackupModel`
**Source:** `pinecone/pinecone.py:1191-1210`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| backup_id | str | Yes | — | The ID of the backup to describe. |

**Returns:** `BackupModel` — An object with full backup details: `backup_id`, `source_index_name`, `source_index_id`, `status`, `cloud`, `region`, `name`, `description`, `dimension`, `metric`, `record_count`, `namespace_count`, `size_bytes`, `created_at`.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `NotFoundException` | The backup with the specified `backup_id` does not exist. |
| `PineconeApiException` | The request failed for other reasons. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

backup = pc.describe_backup(backup_id="backup-123")
print(f"Backup: {backup.name}")
print(f"Status: {backup.status}")
print(f"Source Index: {backup.source_index_name}")
print(f"Created: {backup.created_at}")
print(f"Records: {backup.record_count}")
print(f"Size: {backup.size_bytes} bytes")
print(f"Cloud: {backup.cloud}, Region: {backup.region}")
```

**Notes**

- All arguments must be passed as keyword arguments.
- Use this method to check backup status during the backup creation process.
- Status transitions: "Initializing" → "Ready" or "Failed".

---

### `PineconeAsyncio.describe_backup(*, backup_id: str) -> Awaitable[BackupModel]`

Asynchronous version of `describe_backup()`. Retrieves detailed information about a specific backup.

**Import:** `from pinecone import PineconeAsyncio, BackupModel`
**Source:** `pinecone/pinecone_asyncio.py:1232-1239`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| backup_id | str | Yes | — | The ID of the backup to describe. |

**Returns:** `Awaitable[BackupModel]` — An awaitable that resolves to a `BackupModel` object.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `NotFoundException` | The backup does not exist. |
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

**Example**

```python
import asyncio
from pinecone import PineconeAsyncio

async def describe_backup_async():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    backup = await pc.describe_backup(backup_id="backup-123")
    print(f"{backup.name}: {backup.status}")

asyncio.run(describe_backup_async())
```

**Notes**

- Same behavior as synchronous version.

---

### `Pinecone.delete_backup(*, backup_id: str) -> None`

Deletes a backup permanently.

**Import:** `from pinecone import Pinecone`
**Source:** `pinecone/pinecone.py:1212-1228`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent — repeated calls do not raise an error if the backup is already deleted
**Side effects:** Permanently deletes the backup resource and all associated data

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| backup_id | str | Yes | — | The ID of the backup to delete. |

**Returns:** `None` — This method returns nothing on success.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `NotFoundException` | The backup with the specified `backup_id` does not exist (may only occur on first call; subsequent calls are idempotent). |
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Delete a backup
pc.delete_backup(backup_id="backup-123")
print("Backup deleted successfully")
```

**Notes**

- All arguments must be passed as keyword arguments.
- This operation is permanent and cannot be undone.
- Repeated calls to delete the same backup are safe (idempotent).

---

### `PineconeAsyncio.delete_backup(*, backup_id: str) -> Awaitable[None]`

Asynchronous version of `delete_backup()`. Deletes a backup permanently.

**Import:** `from pinecone import PineconeAsyncio`
**Source:** `pinecone/pinecone_asyncio.py:1241-1248`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** Permanently deletes the backup resource

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| backup_id | str | Yes | — | The ID of the backup to delete. |

**Returns:** `Awaitable[None]` — An awaitable that resolves to `None` on success.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `NotFoundException` | The backup does not exist. |
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

**Example**

```python
import asyncio
from pinecone import PineconeAsyncio

async def delete_backup_async():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")
    await pc.delete_backup(backup_id="backup-123")
    print("Backup deleted")

asyncio.run(delete_backup_async())
```

**Notes**

- Same behavior as synchronous version.

---

### `Pinecone.list_restore_jobs(*, limit: int | None = 10, pagination_token: str | None = None) -> RestoreJobList`

Lists all restore jobs in the project.

**Import:** `from pinecone import Pinecone, RestoreJobList`
**Source:** `pinecone/pinecone.py:1230-1253`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| limit | int \| None | No | `10` | The maximum number of restore jobs to return per page. |
| pagination_token | str \| None | No | `None` | Token for pagination to retrieve subsequent pages. |

**Returns:** `RestoreJobList` — An object containing: `data` (list of `RestoreJobModel` objects), and `pagination` (object with `next` token or `None` if no more pages).

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# List all restore jobs
restore_jobs = pc.list_restore_jobs(limit=20)
print(f"Total restore jobs: {len(restore_jobs.data)}")
for job in restore_jobs.data:
    print(f"  - Job ID: {job.restore_job_id}")
    print(f"    Status: {job.status}")
    print(f"    Source Backup: {job.backup_id}")
    print(f"    Target Index: {job.target_index_name}")
    print(f"    Progress: {job.percent_complete}%")

# Pagination example
next_token = restore_jobs.pagination.next if restore_jobs.pagination else None
if next_token:
    page2 = pc.list_restore_jobs(limit=20, pagination_token=next_token)
    print(f"Page 2 has {len(page2.data)} restore jobs")
```

**Notes**

- All arguments must be passed as keyword arguments.
- Default `limit` is 10. Increase to retrieve more jobs per call.
- This method lists all restore jobs across the entire project, regardless of source backup or target index.

---

### `PineconeAsyncio.list_restore_jobs(*, limit: int | None = 10, pagination_token: str | None = None) -> Awaitable[RestoreJobList]`

Asynchronous version of `list_restore_jobs()`. Lists all restore jobs in the project.

**Import:** `from pinecone import PineconeAsyncio, RestoreJobList`
**Source:** `pinecone/pinecone_asyncio.py:1250-1260`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| limit | int \| None | No | `10` | The maximum number of restore jobs to return. |
| pagination_token | str \| None | No | `None` | Token for pagination. |

**Returns:** `Awaitable[RestoreJobList]` — An awaitable that resolves to a `RestoreJobList` object.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

**Example**

```python
import asyncio
from pinecone import PineconeAsyncio

async def list_restore_jobs_async():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    jobs = await pc.list_restore_jobs(limit=20)
    for job in jobs.data:
        print(f"Job {job.restore_job_id}: {job.status} ({job.percent_complete}%)")

asyncio.run(list_restore_jobs_async())
```

**Notes**

- Same behavior as synchronous version.

---

### `Pinecone.describe_restore_job(*, job_id: str) -> RestoreJobModel`

Retrieves detailed information about a specific restore job, including its progress.

**Import:** `from pinecone import Pinecone, RestoreJobModel`
**Source:** `pinecone/pinecone.py:1255-1274`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| job_id | str | Yes | — | The ID of the restore job to describe. |

**Returns:** `RestoreJobModel` — An object with restore job details: `restore_job_id`, `backup_id`, `target_index_name`, `target_index_id`, `status`, `created_at`, `completed_at` (or `None` if not complete), `percent_complete`.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `NotFoundException` | The restore job with the specified `job_id` does not exist. |
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

job = pc.describe_restore_job(job_id="job-123")
print(f"Restore Job ID: {job.restore_job_id}")
print(f"Status: {job.status}")
print(f"Source Backup: {job.backup_id}")
print(f"Target Index: {job.target_index_name}")
print(f"Progress: {job.percent_complete}%")
print(f"Created: {job.created_at}")
if job.completed_at:
    print(f"Completed: {job.completed_at}")
```

**Notes**

- All arguments must be passed as keyword arguments.
- Use this method to monitor restore job progress.
- Status values: "Initializing", "Running", "Completed", "Failed".
- `percent_complete` ranges from 0 to 100 and indicates progress only for "Running" jobs.
- `completed_at` is `None` until the job finishes.

---

### `PineconeAsyncio.describe_restore_job(*, job_id: str) -> Awaitable[RestoreJobModel]`

Asynchronous version of `describe_restore_job()`. Retrieves detailed information about a specific restore job.

**Import:** `from pinecone import PineconeAsyncio, RestoreJobModel`
**Source:** `pinecone/pinecone_asyncio.py:1262-1269`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| job_id | str | Yes | — | The ID of the restore job to describe. |

**Returns:** `Awaitable[RestoreJobModel]` — An awaitable that resolves to a `RestoreJobModel` object.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `NotFoundException` | The restore job does not exist. |
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

**Example**

```python
import asyncio
from pinecone import PineconeAsyncio

async def describe_restore_job_async():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    job = await pc.describe_restore_job(job_id="job-123")
    print(f"{job.restore_job_id}: {job.status} ({job.percent_complete}%)")

asyncio.run(describe_restore_job_async())
```

**Notes**

- Same behavior as synchronous version.

---

### `Pinecone.create_index_from_backup(*, name: str, backup_id: str, deletion_protection: DeletionProtection | str = "disabled", tags: dict[str, str] | None = None, timeout: int | None = None) -> IndexModel`

Creates a new index by restoring data from a backup.

**Import:** `from pinecone import Pinecone, IndexModel`
**Source:** `pinecone/pinecone.py:658-709`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Not idempotent — repeated calls will create duplicate indexes if the index name differs
**Side effects:** Creates a new index resource in the Pinecone API

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| name | str | Yes | — | The name for the new index to be created. Must be unique within the project. |
| backup_id | str | Yes | — | The ID of the backup to restore from. The backup must be in "Ready" status. |
| deletion_protection | DeletionProtection \| str | No | `"disabled"` | Whether the index should be protected from deletion. One of `"enabled"` or `"disabled"`. Can be changed later with `configure_index()`. |
| tags | dict[str, str] \| None | No | `None` | Optional tags (key-value pairs) to attach to the index for organization and identification. |
| timeout | int \| None | No | `None` | Seconds to wait for index readiness. If `None`, wait indefinitely; if `>=0`, time out after this many seconds; if `-1`, return immediately without waiting. |

**Returns:** `IndexModel` — An object describing the created index with fields: `name`, `dimension`, `metric`, `host`, `status`, `ready`, `spec`, `deletion_protection`, `tags`, `created_at`.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The request failed. May occur if the `backup_id` is invalid, if the backup does not exist, if the backup is not in "Ready" status, if an index with the same `name` already exists, or if the API call fails. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# List available backups
backups = pc.list_backups(limit=10)
if backups.data:
    backup_id = backups.data[0].backup_id

    # Create an index from the backup
    index = pc.create_index_from_backup(
        name="restored_index",
        backup_id=backup_id,
        deletion_protection="disabled",
        tags={"restore_date": "2024-03-17"}
    )

    print(f"Index created: {index.name}")
    print(f"Status: {index.status}")
    print(f"Ready: {index.ready}")
```

**Notes**

- All arguments must be passed as keyword arguments.
- The new index is created with the same vector dimension and metric as the backup.
- Index creation is asynchronous; use `describe_index()` to poll until `ready=True`.
- Data from the backup is restored after the index creation completes.
- The restore operation creates a RestoreJob that can be monitored with `describe_restore_job()`.

---

### `PineconeAsyncio.create_index_from_backup(*, name: str, backup_id: str, deletion_protection: DeletionProtection | str = "disabled", tags: dict[str, str] | None = None, timeout: int | None = None) -> Awaitable[IndexModel]`

Asynchronous version of `create_index_from_backup()`. Creates a new index by restoring data from a backup.

**Import:** `from pinecone import PineconeAsyncio, IndexModel`
**Source:** `pinecone/pinecone_asyncio.py:722-755`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Not idempotent
**Side effects:** Creates a new index resource in the Pinecone API

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| name | str | Yes | — | The name for the new index to be created. |
| backup_id | str | Yes | — | The ID of the backup to restore from. |
| deletion_protection | DeletionProtection \| str | No | `"disabled"` | Whether the index should be protected from deletion. |
| tags | dict[str, str] \| None | No | `None` | Optional tags to attach to the index. |
| timeout | int \| None | No | `None` | Seconds to wait for index readiness. |

**Returns:** `Awaitable[IndexModel]` — An awaitable that resolves to an `IndexModel` object describing the created index.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The request failed. May occur if the backup does not exist or if an index with the same name already exists. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

**Example**

```python
import asyncio
from pinecone import PineconeAsyncio

async def restore_from_backup():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    # Get backup ID
    backups = await pc.list_backups(limit=10)
    if backups.data:
        backup_id = backups.data[0].backup_id

        # Restore from backup
        index = await pc.create_index_from_backup(
            name="restored_index",
            backup_id=backup_id
        )
        print(f"Index created: {index.name}")

asyncio.run(restore_from_backup())
```

**Notes**

- Same behavior as synchronous version, but returns an awaitable for async/await usage.

---

## Data Models

### BackupModel

Represents a backup of an index.

**Import:** `from pinecone import BackupModel`
**Source:** `pinecone/core/openapi/db_control/model/backup_model.py:50-387`

| Field | Type | Description |
|-------|------|-------------|
| backup_id | str | Unique identifier for the backup. |
| source_index_name | str | Name of the index from which the backup was taken. |
| source_index_id | str | ID of the source index. |
| status | str | Current status: "Initializing", "Ready", or "Failed". |
| cloud | str | Cloud provider where the backup is stored (e.g., "aws", "gcp", "azure"). |
| region | str | Cloud region where the backup is stored. |
| name | str | User-defined name for the backup. |
| description | str | User-provided description. |
| dimension | int | Vector dimension of the backup. |
| metric | str | Distance metric: "cosine", "euclidean", or "dotproduct". |
| record_count | int | Total number of records in the backup. |
| namespace_count | int | Number of namespaces in the backup. |
| size_bytes | int | Size of the backup in bytes. |
| created_at | str | ISO 8601 timestamp of backup creation. |

---

### BackupList

Container for paginated backup list results.

**Import:** `from pinecone import BackupList`
**Source:** `pinecone/core/openapi/db_control/model/backup_list.py:50-302`

| Field | Type | Description |
|-------|------|-------------|
| data | list[BackupModel] | List of backup objects. |
| pagination | PaginationResponse \| None | Pagination info; contains `next` token or `None` if no more pages. |

---

### RestoreJobModel

Represents a restore job that restores a backup to a new or existing index.

**Import:** `from pinecone import RestoreJobModel`
**Source:** `pinecone/core/openapi/db_control/model/restore_job_model.py:36-347`

| Field | Type | Description |
|-------|------|-------------|
| restore_job_id | str | Unique identifier for the restore job. |
| backup_id | str | ID of the backup being restored. |
| target_index_name | str | Name of the index being restored into. |
| target_index_id | str | ID of the target index. |
| status | str | Current status: "Initializing", "Running", "Completed", or "Failed". |
| created_at | datetime | Timestamp when the restore job started. |
| completed_at | datetime \| None | Timestamp when the restore job finished; `None` if not yet complete. |
| percent_complete | float | Progress from 0 to 100. Valid only when status is "Running". |

---

### RestoreJobList

Container for paginated restore job list results.

**Import:** `from pinecone import RestoreJobList`
**Source:** `pinecone/core/openapi/db_control/model/restore_job_list.py:50-308`

| Field | Type | Description |
|-------|------|-------------|
| data | list[RestoreJobModel] | List of restore job objects. |
| pagination | PaginationResponse \| None | Pagination info; contains `next` token or `None` if no more pages. |

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

### Polling for restore job completion

```python
import time
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Assume restore job has been created via create_index_from_backup()
job_id = "job-123"

while True:
    job = pc.describe_restore_job(job_id=job_id)
    print(f"Progress: {job.percent_complete}%")

    if job.status == "Completed":
        print("Restore completed successfully")
        break
    elif job.status == "Failed":
        raise Exception("Restore job failed")
    else:
        time.sleep(10)
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

## Notes and Warnings

- **Keyword-only arguments:** All methods use the `@require_kwargs` decorator, requiring all arguments (except `self`) to be passed as keyword arguments. Positional arguments will raise a `TypeError`.
- **Async support:** `PineconeAsyncio` provides async/await versions of all methods for non-blocking operations.
- **Backup lifecycle:** Backups transition through states: "Initializing" → "Ready" (or "Failed"). Only "Ready" backups can be used for restore operations.
- **Restore lifecycle:** Restore jobs transition through: "Initializing" → "Running" → "Completed" (or "Failed").
- **Data durability:** Backups are immutable once created; they preserve the exact state of the index at creation time.
- **Pricing:** Backups incur storage costs. Regularly clean up old backups using `delete_backup()`.
