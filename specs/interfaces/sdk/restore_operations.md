# Restore Operations

Restore operations enable you to create new indexes from previously created backups and track the progress of restore jobs. These operations allow data recovery and index replication workflows.

---

## `Pinecone.create_index_from_backup()`

Creates a new index by restoring data from a backup.

**Source:** `pinecone/pinecone.py:658-709`, `pinecone/db_control/resources/sync/index.py:151-183`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent — repeated calls will create duplicate indexes if the index name differs
**Side effects:** Creates a new index; initiates a restore job that copies data from the backup.

### Signature

```python
def create_index_from_backup(
    self,
    *,
    name: str,
    backup_id: str,
    deletion_protection: (DeletionProtection | str) | None = "disabled",
    tags: dict[str, str] | None = None,
    timeout: int | None = None
) -> IndexModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name for the new index to create. Must be unique within the project. |
| `backup_id` | `string` | Yes | — | v1.0 | No | The ID of the backup to restore. Obtain this from `list_backups()` or `describe_backup()`. The backup must be in `"Ready"` status. |
| `deletion_protection` | `(DeletionProtection \| string) \| None` | No | `"disabled"` | v1.0 | No | Whether the index can be deleted. When `"enabled"`, `delete_index()` will fail unless deletion protection is first disabled with `configure_index()`. |
| `tags` | `dict[str, str]` | No | `None` | v1.0 | No | Key-value pairs to attach to the index for organization and identification. When omitted, the index is created with no tags. |
| `timeout` | `integer (int32)` | No | `None` | v1.0 | No | Number of seconds to wait for the index to reach `"ready"` status. If `None`, wait indefinitely. If `-1`, return immediately without polling and caller must use `describe_index()` to check readiness. If `>= 0`, raise `TimeoutError` if the index is not ready within this duration. Polls every 5 seconds. |

### Returns

**Type:** `IndexModel` — The newly created index when ready. The model contains the complete index configuration including name, dimension, metric, host, status, ready, spec, deletion_protection, tags, and created_at. When `timeout=-1`, returns the index in its current state (may not be `"ready"` yet).

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `name` is an empty string or contains invalid characters. |
| `NotFoundException` | The backup ID does not exist or is inaccessible. |
| `PineconeApiException` | The request failed. May occur if the `backup_id` is invalid, if the backup is not in `"Ready"` status, if an index with the same `name` already exists, or if the API call fails. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TimeoutError` | The index did not reach `"ready"` status within the specified `timeout`. |
| `Exception` | The index fails to initialize (status becomes `InitializationFailed`). |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

### Behavior

- The index creation initiates immediately upon calling this method. The returned `IndexModel` has a status that transitions from `"Initializing"` to `"Ready"` as the restore completes.
- The source backup is not modified or deleted by this operation.
- The created index inherits the vector dimension and metric from the source backup.
- Index tags from the backup are not automatically carried forward; only the tags explicitly passed in `tags` parameter are set.
- If the index name is not unique within the project, a conflict error occurs.
- Repeated calls with the same parameters will create multiple indexes (non-idempotent).
- The restore operation creates a RestoreJob that can be monitored with `describe_restore_job()`.
- Data from the backup is restored after the index creation completes.

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# List available backups to find one to restore
backups = pc.list_backups(limit=10)
if backups.data:
    backup_id = backups.data[0].backup_id

    # Create a new index from the backup
    restored_index = pc.create_index_from_backup(
        name="my-restored-index",
        backup_id=backup_id,
        deletion_protection="disabled",
        tags={"environment": "production", "source": "backup"}
    )

    print(f"Index created: {restored_index.name}")
    print(f"Status: {restored_index.status}")
    print(f"Ready: {restored_index.ready}")
```

### Notes

- All arguments must be passed as keyword arguments.
- Creation does not block indefinitely by default. If you need to ensure the index is ready before proceeding, use the default `timeout=None` to wait indefinitely, or specify a timeout value.
- When `timeout=-1`, the method returns immediately. Check the index status periodically using `describe_index()` until `status.ready` is `true`.
- The created index starts with no data until the restore job completes, which can take several minutes for large backups.

---

## `PineconeAsyncio.create_index_from_backup()`

Asynchronous version of `create_index_from_backup()`. Creates a new index by restoring data from a backup.

**Source:** `pinecone/pinecone_asyncio.py:722-755`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent
**Side effects:** Creates a new index resource in the Pinecone API

### Signature

```python
async def create_index_from_backup(
    self,
    *,
    name: str,
    backup_id: str,
    deletion_protection: (DeletionProtection | str) | None = "disabled",
    tags: dict[str, str] | None = None,
    timeout: int | None = None
) -> IndexModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name for the new index to be created. |
| `backup_id` | `string` | Yes | — | v1.0 | No | The ID of the backup to restore from. |
| `deletion_protection` | `(DeletionProtection \| string) \| None` | No | `"disabled"` | v1.0 | No | Whether the index should be protected from deletion. |
| `tags` | `dict[str, str]` | No | `None` | v1.0 | No | Optional tags to attach to the index. |
| `timeout` | `integer (int32)` | No | `None` | v1.0 | No | Seconds to wait for index readiness. |

### Returns

**Type:** `Awaitable[IndexModel]` — An awaitable that resolves to an `IndexModel` object describing the created index.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. May occur if the backup does not exist or if an index with the same name already exists. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

### Example

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

### Notes

- Same behavior as synchronous version, but returns an awaitable for async/await usage.

---

## `Pinecone.list_restore_jobs()`

Lists all restore jobs in the project, with pagination support.

**Source:** `pinecone/pinecone.py:1230-1253`, `pinecone/db_control/resources/sync/restore_job.py:58-74`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

### Signature

```python
def list_restore_jobs(
    self,
    *,
    limit: int | None = 10,
    pagination_token: str | None = None
) -> RestoreJobList
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `limit` | `integer (int32, 1–100)` | No | `10` | v1.0 | No | Maximum number of restore jobs to return in a single page. Requests with `limit > 100` are clamped to `100`. |
| `pagination_token` | `string` | No | `None` | v1.0 | No | The pagination token from the previous page. When omitted, returns the first page of restore jobs ordered by most recent creation first. |

### Returns

**Type:** `RestoreJobList` — A container object with a `data` field containing an array of `RestoreJobModel` objects and a `pagination` field for fetching subsequent pages.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. May occur if the `pagination_token` is invalid or expired. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

### Behavior

- Returns restore jobs for all backups in the project, not filtered by backup or time range.
- Results are ordered by creation time, most recent first.
- When no restore jobs exist, returns an empty list (empty `data` array).
- Deleted or expired restore jobs are not included in results.
- The `limit` parameter is clamped: requests for `limit > 100` are treated as `limit=100`.
- When `limit` is omitted or `None`, defaults to `10`.
- The `pagination` field is present in every response. When there are no more pages, `pagination.next` is `null` or omitted.

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# List all restore jobs, starting with the most recent
all_jobs = []
pagination_token = None

while True:
    job_list = pc.list_restore_jobs(limit=20, pagination_token=pagination_token)
    all_jobs.extend(job_list.data)

    # Check if there are more pages
    if job_list.pagination and job_list.pagination.next:
        pagination_token = job_list.pagination.next
    else:
        break

print(f"Total restore jobs: {len(all_jobs)}")
for job in all_jobs:
    print(f"  - Job ID: {job.restore_job_id}")
    print(f"    Status: {job.status}")
    print(f"    Source Backup: {job.backup_id}")
    print(f"    Target Index: {job.target_index_name}")
    print(f"    Progress: {job.percent_complete}%")
```

### Notes

- All arguments must be passed as keyword arguments.
- Default `limit` is 10. Increase to retrieve more jobs per call.
- This method lists all restore jobs across the entire project, regardless of source backup or target index.
- Restore jobs are retained for a fixed period after completion or failure; very old jobs may no longer appear in results.
- The list reflects the current state of jobs at query time; subsequent calls may return different results if jobs complete or fail between calls.

---

## `PineconeAsyncio.list_restore_jobs()`

Asynchronous version of `list_restore_jobs()`. Lists all restore jobs in the project.

**Source:** `pinecone/pinecone_asyncio.py:1250-1260`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

### Signature

```python
async def list_restore_jobs(
    self,
    *,
    limit: int | None = 10,
    pagination_token: str | None = None
) -> RestoreJobList
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `limit` | `integer (int32)` | No | `10` | v1.0 | No | The maximum number of restore jobs to return. |
| `pagination_token` | `string` | No | `None` | v1.0 | No | Token for pagination. |

### Returns

**Type:** `Awaitable[RestoreJobList]` — An awaitable that resolves to a `RestoreJobList` object.

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

async def list_restore_jobs_async():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    jobs = await pc.list_restore_jobs(limit=20)
    for job in jobs.data:
        print(f"Job {job.restore_job_id}: {job.status} ({job.percent_complete}%)")

asyncio.run(list_restore_jobs_async())
```

### Notes

- Same behavior as synchronous version.

---

## `Pinecone.describe_restore_job()`

Retrieves detailed information about a specific restore job by ID, including its progress.

**Source:** `pinecone/pinecone.py:1255-1274`, `pinecone/db_control/resources/sync/restore_job.py:46-56`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

### Signature

```python
def describe_restore_job(self, *, job_id: str) -> RestoreJobModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `job_id` | `string` | Yes | — | v1.0 | No | The ID of the restore job to describe. Obtain this from `list_restore_jobs()` or a previous `create_index_from_backup()` call. |

### Returns

**Type:** `RestoreJobModel` — An object with complete information about the restore job, including its ID, status, source backup ID, target index details, and progress.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The job ID does not exist or has expired. |
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments instead of keyword arguments. |

### Behavior

- Returns information about the restore job regardless of its current status (running, completed, failed).
- The `status` field indicates the current state: `"Initializing"`, `"Running"`, `"Completed"`, or `"Failed"`.
- The `percent_complete` field indicates progress as a percentage (0-100). When `status="Completed"`, `percent_complete` is `100`. When `status="Failed"`, the value indicates how much data was copied before failure.
- `completed_at` is `None` until the job finishes.

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Describe a specific restore job
job = pc.describe_restore_job(job_id="job-abc123")

print(f"Restore Job ID: {job.restore_job_id}")
print(f"Status: {job.status}")
print(f"Source Backup: {job.backup_id}")
print(f"Target Index: {job.target_index_name}")
print(f"Progress: {job.percent_complete}%")
print(f"Created at: {job.created_at}")

if job.completed_at:
    print(f"Completed at: {job.completed_at}")
```

### Notes

- All arguments must be passed as keyword arguments.
- Use this method to monitor the progress of a restore operation initiated by `create_index_from_backup()`.
- Failed restore jobs retain their status and details for the retention period to allow debugging; the target index may be in an inconsistent state and should not be used.

---

## `PineconeAsyncio.describe_restore_job()`

Asynchronous version of `describe_restore_job()`. Retrieves detailed information about a specific restore job.

**Source:** `pinecone/pinecone_asyncio.py:1262-1269`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None — read-only operation

### Signature

```python
async def describe_restore_job(self, *, job_id: str) -> RestoreJobModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `job_id` | `string` | Yes | — | v1.0 | No | The ID of the restore job to describe. |

### Returns

**Type:** `Awaitable[RestoreJobModel]` — An awaitable that resolves to a `RestoreJobModel` object.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The restore job does not exist. |
| `PineconeApiException` | The request failed. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `TypeError` | Arguments are passed as positional arguments. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def describe_restore_job_async():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    job = await pc.describe_restore_job(job_id="job-123")
    print(f"{job.restore_job_id}: {job.status} ({job.percent_complete}%)")

asyncio.run(describe_restore_job_async())
```

### Notes

- Same behavior as synchronous version.

---

## Common Patterns

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

---

## Data Models

### `RestoreJobModel`

Represents a restore job that has been initiated or completed.

**Source:** `pinecone/core/openapi/db_control/model/restore_job_model.py:36-127`, `pinecone/db_control/models/restore_job_model.py:8-26`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `restore_job_id` | `string` | No | v1.0 | No | Unique identifier for the restore job. |
| `backup_id` | `string` | No | v1.0 | No | The ID of the backup being restored. |
| `target_index_name` | `string` | No | v1.0 | No | The name of the index into which data is being restored. |
| `target_index_id` | `string` | No | v1.0 | No | The internal ID of the target index. |
| `status` | `string (enum: Initializing, Running, Completed, Failed)` | No | v1.0 | No | Current status of the restore job. Transitions from `Initializing` -> `Running` -> (`Completed` or `Failed`). |
| `created_at` | `string (date-time)` | No | v1.0 | No | ISO 8601 timestamp when the restore job was initiated. |
| `completed_at` | `string (date-time)` | Yes | v1.0 | No | ISO 8601 timestamp when the restore job finished (either successfully or with failure). `null` when the job is still `Running` or `Initializing`. |
| `percent_complete` | `number (double, 0–100)` | Yes | v1.0 | No | Progress of the restore as a percentage. Valid only when status is `"Running"`. |

#### `to_dict()`

Converts the RestoreJobModel to a dictionary representation.

**Source:** `pinecone/db_control/models/restore_job_model.py:24-25`

**Returns:** `dict` — A dictionary containing all fields and values of the restore job.

**Example**

```python
from pinecone import Pinecone

pc = Pinecone()

# Get a restore job and convert to dictionary
job = pc.describe_restore_job(job_id="job-abc123")
job_dict = job.to_dict()

print(job_dict)
# Output: {
#   "restore_job_id": "job-abc123",
#   "backup_id": "backup-xyz789",
#   "target_index_name": "my-restored-index",
#   "target_index_id": "idx-def456",
#   "status": "in_progress",
#   "created_at": "2025-03-17T10:30:00Z",
#   "completed_at": None,
#   "percent_complete": 45.5
# }
```

### `RestoreJobList`

A container for a paginated list of restore jobs.

**Source:** `pinecone/db_control/models/restore_job_list.py:17-50`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `data` | `array of RestoreJobModel` | No | v1.0 | No | The restore jobs in the current page. |
| `pagination` | `object (PaginationResponse)` | No | v1.0 | No | Pagination metadata including `next` token for fetching the subsequent page. |

The `PaginationResponse` object has:

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `next` | `string` | Yes | The pagination token to fetch the next page of results. `null` or omitted if there are no more pages. |
