# Restore Operations

Restore operations enable you to create new indexes from previously created backups and track the progress of restore jobs. These operations allow data recovery and index replication workflows.

---

## `Pinecone.create_index_from_backup()`

Creates a new index by restoring data from a backup.

**Source:** `pinecone/pinecone.py:658-709`, `pinecone/db_control/resources/sync/index.py:151-183`

**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent
**Side effects:** Creates a new index; initiates a restore job that copies data from the backup.

### Signature

```python
def create_index_from_backup(
    self,
    *,
    name: str,
    backup_id: str,
    deletion_protection: DeletionProtection | str | None = "disabled",
    tags: dict[str, str] | None = None,
    timeout: int | None = None
) -> IndexModel
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | `string` | Yes | — | The name for the new index to create. Must be unique within the project. |
| `backup_id` | `string` | Yes | — | The ID of the backup to restore. Obtain this from `list_backups()` or `describe_backup()`. |
| `deletion_protection` | `string (enum: enabled, disabled)` | No | `"disabled"` | Whether the index can be deleted. When `"enabled"`, `delete_index()` will fail unless deletion protection is first disabled with `configure_index()`. |
| `tags` | `dict[str, str]` | No | `None` | Key-value pairs to attach to the index for organization and identification. When omitted, the index is created with no tags. |
| `timeout` | `integer (int32)` | No | `None` | Number of seconds to wait for the index to reach `"ready"` status. If `None`, wait indefinitely. If `-1`, return immediately without polling and caller must use `describe_index()` to check readiness. If `>= 0`, raise `TimeoutError` if the index is not ready within this duration. Polls every 5 seconds. |

### Returns

**Type:** `IndexModel` — The newly created index when ready. The model contains the complete index configuration including name, dimension, metric, status, and spec. When `timeout=-1`, returns the index in its current state (may not be `"ready"` yet).

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `name` is an empty string or contains invalid characters. |
| `NotFoundException` | The backup ID does not exist or is inaccessible. |
| `pinecone.PineconeApiException` | The backup is in `"initialized"` or `"in_progress"` status and cannot be restored yet. |
| `TimeoutError` | The index did not reach `"ready"` status within the specified `timeout`. |
| `Exception` | The index fails to initialize (status becomes `InitializationFailed`). |

### Behavior

- The index creation initiates immediately upon calling this method. The returned `IndexModel` has a status that transitions from `"Initializing"` to `"Ready"` as the restore completes.
- The source backup is not modified or deleted by this operation.
- The created index inherits the vector dimension and metric from the source backup.
- Index tags from the backup are not automatically carried forward; only the tags explicitly passed in `tags` parameter are set.
- If the index name is not unique within the project, a conflict error occurs.
- Repeated calls with the same parameters will create multiple indexes (non-idempotent).

### Example

```python
from pinecone import Pinecone

pc = Pinecone()

# List available backups to find one to restore
backups = pc.list_backups()
if backups:
    backup_id = backups[0].backup_id

    # Create a new index from the backup
    restored_index = pc.create_index_from_backup(
        name="my-restored-index",
        backup_id=backup_id,
        deletion_protection="disabled",
        tags={"environment": "production", "source": "backup"}
    )

    print(f"Index created: {restored_index.name}")
    print(f"Status: {restored_index.status.state}")
    print(f"Ready: {restored_index.status.ready}")
```

### Notes

- Creation does not block indefinitely by default. If you need to ensure the index is ready before proceeding, use the default `timeout=None` to wait indefinitely, or specify a timeout value.
- When `timeout=-1`, the method returns immediately. Check the index status periodically using `describe_index()` until `status.ready` is `true`.
- The created index starts with no data until the restore job completes, which can take several minutes for large backups.

---

## `Pinecone.list_restore_jobs()`

Lists all restore jobs in the project, with pagination support.

**Source:** `pinecone/pinecone.py:1231-1253`, `pinecone/db_control/resources/sync/restore_job.py:58-74`

**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None

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

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | `integer (int32, 1–100)` | No | `10` | Maximum number of restore jobs to return in a single page. Requests with `limit > 100` are clamped to `100`. |
| `pagination_token` | `string` | No | `None` | The pagination token from the previous page. When omitted, returns the first page of restore jobs ordered by most recent creation first. |

### Returns

**Type:** `RestoreJobList` — A container object with a `data` field containing an array of `RestoreJobModel` objects and a `pagination` field for fetching subsequent pages.

### Raises

| Exception | Condition |
|-----------|-----------|
| `pinecone.PineconeApiException` | The `pagination_token` is invalid or expired. |

### Behavior

- Returns restore jobs for all backups in the project, not filtered by backup or time range.
- Results are ordered by creation time, most recent first.
- When no restore jobs exist, returns an empty list (empty `data` array).
- Deleted or expired restore jobs are not included in results.
- The `limit` parameter is clamped: requests for `limit > 100` are treated as `limit=100`.
- When `limit` is omitted or `None`, defaults to `10`.
- The `pagination` field is present in every response. When there are no more pages, `pagination.next` is `null` or omitted.

### Pagination

The returned `RestoreJobList` has a `pagination` attribute:
- If `pagination.next` is a non-empty string, more results are available. Pass this token to a subsequent call to fetch the next page.
- If `pagination.next` is `None` or omitted, no further pages exist.

### Example

```python
from pinecone import Pinecone

pc = Pinecone()

# List all restore jobs, starting with the most recent
all_jobs = []
pagination_token = None

while True:
    job_list = pc.list_restore_jobs(limit=20, pagination_token=pagination_token)
    all_jobs.extend(job_list)

    # Check if there are more pages
    if job_list.pagination and job_list.pagination.next:
        pagination_token = job_list.pagination.next
    else:
        break

print(f"Total restore jobs: {len(all_jobs)}")
for job in all_jobs:
    print(f"Job ID: {job.restore_job_id}, Status: {job.status}")
```

### Notes

- Restore jobs are retained for a fixed period after completion or failure; very old jobs may no longer appear in results.
- The list reflects the current state of jobs at query time; subsequent calls may return different results if jobs complete or fail between calls.

---

## `Pinecone.describe_restore_job()`

Retrieves detailed information about a specific restore job by ID.

**Source:** `pinecone/pinecone.py:1256-1274`, `pinecone/db_control/resources/sync/restore_job.py:46-56`

**Added:** v1.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** None

### Signature

```python
def describe_restore_job(self, *, job_id: str) -> RestoreJobModel
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `job_id` | `string` | Yes | — | The ID of the restore job to describe. Obtain this from `list_restore_jobs()` or a previous `create_index_from_backup()` call. |

### Returns

**Type:** `RestoreJobModel` — An object with complete information about the restore job, including its ID, status, source backup ID, target index details, and progress.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The job ID does not exist or has expired. |

### Behavior

- Returns information about the restore job regardless of its current status (running, completed, failed).
- The `status` field indicates the current state: `"initialized"`, `"in_progress"`, `"completed"`, or `"failed"`.
- The `percent_complete` field indicates progress as a percentage (0–100). When `status="completed"`, `percent_complete` is `100`. When `status="failed"`, the value indicates how much data was copied before failure.

### Example

```python
from pinecone import Pinecone

pc = Pinecone()

# Describe a specific restore job
job = pc.describe_restore_job(job_id="job-abc123")

print(f"Restore Job ID: {job.restore_job_id}")
print(f"Status: {job.status}")
print(f"Source Backup: {job.backup_id}")
print(f"Target Index: {job.target_index_name}")
print(f"Progress: {job.percent_complete}%")
print(f"Created at: {job.created_at}")

if job.status == "completed":
    print(f"Completed at: {job.completed_at}")
```

### Notes

- Use this method to monitor the progress of a restore operation initiated by `create_index_from_backup()`.
- Failed restore jobs retain their status and details for the retention period to allow debugging; the target index may be in an inconsistent state and should not be used.

---

## Data Models

### `RestoreJobModel`

Represents a restore job that has been initiated or completed.

**Source:** `pinecone/core/openapi/db_control/model/restore_job_model.py:36-127`, `pinecone/db_control/models/restore_job_model.py:8-26`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `restore_job_id` | `string` | No | — | No | Unique identifier for the restore job. |
| `backup_id` | `string` | No | — | No | The ID of the backup being restored. |
| `target_index_name` | `string` | No | — | No | The name of the index into which data is being restored. |
| `target_index_id` | `string` | No | — | No | The internal ID of the target index. |
| `status` | `string (enum: initialized, in_progress, completed, failed)` | No | — | No | Current status of the restore job. Transitions from `initialized` → `in_progress` → (`completed` or `failed`). |
| `created_at` | `string (date-time)` | No | — | No | ISO 8601 timestamp when the restore job was initiated. |
| `completed_at` | `string (date-time)` | Yes | — | No | ISO 8601 timestamp when the restore job finished (either successfully or with failure). Omitted from response when the job is still `in_progress` or `initialized`. |
| `percent_complete` | `number (double, 0–100)` | Yes | — | No | Progress of the restore as a percentage. Omitted when the job status is `initialized` or `failed`. |

#### `to_dict()`

Converts the RestoreJobModel to a dictionary representation.

**Source:** `pinecone/db_control/models/restore_job_model.py:24-25`

**Returns:** `dict` — A dictionary containing all fields and values of the restore job.

### `RestoreJobList`

A container for a paginated list of restore jobs.

**Source:** `pinecone/db_control/models/restore_job_list.py:17-50`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `data` | `array of RestoreJobModel` | No | — | No | The restore jobs in the current page. |
| `pagination` | `object (PaginationResponse)` | No | — | No | Pagination metadata including `next` token for fetching the subsequent page. |

The `PaginationResponse` object has:

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `next` | `string` | Yes | The pagination token to fetch the next page of results. `null` or omitted if there are no more pages. |

