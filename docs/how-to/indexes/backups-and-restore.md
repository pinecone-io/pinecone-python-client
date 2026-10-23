# Backups and restore

Backups are point-in-time snapshots of a managed (serverless) or BYOC index. Use them to
protect against data loss, create copies of an index, or restore a previous state.

Pod-based indexes cannot be backed up. Their snapshot mechanism is a collection, which has
a different API. Restoring a collection needs an API version before `2026-07`, and this SDK
cannot do it. See [collections](../collections.md#create-an-index-from-a-collection).

## Create a backup

Pass the name of the index you want to back up, and optionally a name and description:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

backup = pc.backups.create(
    index_name="product-search",
    name="pre-reindex-snapshot",
    description="Backup before schema migration",
)
print(backup.backup_id)   # e.g. "bk-abc123"
print(backup.status)      # "Initializing"
```

{meth}`~pinecone.client.backups.Backups.create` returns as soon as the server accepts
the request. `status` is `"Initializing"` while the snapshot is built, then `"Ready"` or
`"InitializationFailed"`. There is no `timeout=` to wait on, so poll
{meth}`~pinecone.client.backups.Backups.describe` until the status leaves
`"Initializing"`.

`pc.indexes.create_backup("product-search", name="pre-reindex-snapshot")` is the same
operation under the indexes namespace.


## List backups

{meth}`~pinecone.client.backups.Backups.list` returns a
{class}`~pinecone.models.backups.list.BackupList` holding **one page**, not every backup
in the project. To list all backups, follow the pagination token:

```python
page = pc.backups.list(limit=100)
backups = {b.backup_id: b for b in page}
while page.pagination and page.pagination.next:
    page = pc.backups.list(pagination_token=page.pagination.next)
    backups.update({b.backup_id: b for b in page})

for backup in backups.values():
    print(backup.backup_id, backup.name, backup.status)
```

The token names an offset, not a stable cursor. Backups created or deleted while you
walk the pages shift later offsets, so a walk can skip a backup or return one twice.
Keying by `backup_id`, as above, removes the duplicates. The token already carries the
page size, so the SDK drops `limit` from any call that passes a token.

For every backup of one index, use
{meth}`~pinecone.client.indexes.Indexes.list_backups`, which walks the pages for you:

```python
for backup in pc.indexes.list_backups("product-search"):
    print(backup.backup_id, backup.status)
```

`pc.backups.list(index_name="product-search")` also filters by index, but returns one
page like the project-wide call.


### Backups of a deleted index

A backup outlives its source index. An index-scoped listing looks up the *active* index
with that name, so if every index that used the name has been deleted, the server
answers **404, not an empty list**. Pass `include_deleted=True` to cover every index
that has ever used the name:

```python
page = pc.backups.list(index_name="product-search", include_deleted=True, limit=100)

for backup in page:
    if backup.source_index_deleted_at:
        print(backup.backup_id, "source deleted at", backup.source_index_deleted_at)
```

A 404 *with* `include_deleted=True` means the name has never been used in this project.
The project-wide `pc.backups.list()` already includes backups of deleted indexes, and
passing `include_deleted` without `index_name` raises
{exc}`~pinecone.errors.exceptions.PineconeValueError`.


## Describe a backup

```python
backup = pc.backups.describe(backup_id="bk-abc123")
print(backup.source_index_name)
print(backup.status)            # "Initializing" | "Ready" | "InitializationFailed"
print(backup.record_count)
print(backup.size_bytes)

if backup.schema is not None:
    print(backup.schema.fields["your_vector"].metric)
```

`schema` is an {class}`~pinecone.models.indexes.schema.IndexSchema`, the same model
`pc.indexes.describe(...).schema` returns. It is `None` when the server returns no
schema for the backup, so check it before reading fields. `dense_dimension` is a
shortcut that reports the dimension when the schema has exactly one `dense_vector`
field, and `None` otherwise.


## Restore a backup to a new index

{meth}`~pinecone.Pinecone.create_index_from_backup` restores a backup into a new index.
It is the only way to restore; {meth}`~pinecone.client.indexes.Indexes.create` does not
accept a backup ID.

```python
index = pc.create_index_from_backup(
    name="product-search-restored",
    backup_id="bk-abc123",
    deletion_protection="enabled",
    tags={"env": "production", "team": "search"},
)
print(index.name, index.status.state)
```

The call polls until the new index is ready, for 300 seconds by default, then raises
{exc}`~pinecone.errors.exceptions.PineconeTimeoutError`. The timeout does not cancel
the restore, which continues on the server; monitor it with the restore job below. Pass
a larger `timeout` to wait longer, or `timeout=-1` to return as soon as the restore
starts:

```python
result = pc.create_index_from_backup(
    name="product-search-restored",
    backup_id="bk-abc123",
    timeout=-1,
)
print(result.restore_job_id)
```

### What a restore does and does not carry over

- **Tags:** tags you pass replace the backup's tags. Omit them and the restored index
  gets the backup's tags.
- **Deletion protection:** never inherited. It defaults to `"disabled"`, so an index
  restored from a protected source comes back unprotected unless you pass
  `deletion_protection="enabled"`.
- **Read capacity:** omit `read_capacity` for on-demand capacity, or pass a dedicated
  configuration to bring the index up on dedicated read nodes in the same call. The
  server rejects a dedicated configuration too small to hold the backup.

```python
index = pc.create_index_from_backup(
    name="product-search-restored",
    backup_id="bk-abc123",
    read_capacity={
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {"shards": 2, "replicas": 2},
        },
    },
)
```

The backup must be finished before you restore it. Restoring a backup whose `status` is
not yet `"Ready"` fails with {exc}`~pinecone.errors.exceptions.FailedPreconditionError`
(412), and the message says the backup "is not completed".


## Monitor restore jobs

Each call to `create_index_from_backup` starts a restore job. Describe one by ID:

```python
job = pc.restore_jobs.describe(job_id="rj-abc123")
print(job.target_index_name)
print(job.status)         # "Pending" | "Completed" | "Failed" | "Cancelled"
print(job.completed_at)
```

`describe` returns a {class}`~pinecone.models.backups.model.RestoreJobModel`. A restore
that is running reports `"Pending"`; there is no in-progress state, and
`percent_complete` is `100` once the job completes and `None` before that. Poll for a
terminal status:

```python
import time

job = pc.restore_jobs.describe(job_id="rj-abc123")
while job.status == "Pending":
    time.sleep(5)
    job = pc.restore_jobs.describe(job_id="rj-abc123")
```

```{warning}
A {exc}`~pinecone.errors.exceptions.NotFoundError` from `restore_jobs.describe` does not
prove the job does not exist. The server also answers 404 on a transient failure and for
a job whose target index has been deleted. Treat a 404 as possibly transient unless you
know the ID is wrong.
```

{meth}`~pinecone.client.restore_jobs.RestoreJobs.list` returns a
{class}`~pinecone.models.backups.list.RestoreJobList` holding one page. Follow the token
to read more:

```python
page = pc.restore_jobs.list(limit=100)
jobs = {j.restore_job_id: j for j in page}
while page.pagination and page.pagination.next:
    page = pc.restore_jobs.list(pagination_token=page.pagination.next)
    jobs.update({j.restore_job_id: j for j in page})
```

```{warning}
The restore-job listing is not exhaustive. Pages can overlap, pagination can end while
jobs remain, and jobs whose target index has been deleted are left out. Key the results
by `restore_job_id` as above, and never conclude a job does not exist because the
listing omits it. To track a specific restore, keep the `restore_job_id` that
`create_index_from_backup(..., timeout=-1)` returns and describe it.
```


## Delete a backup

```python
pc.backups.delete(backup_id="bk-abc123")
```

Deleting a backup does not affect the source index or any index restored from it. A
backup with a restore job still running cannot be deleted: the server answers with
{exc}`~pinecone.errors.exceptions.FailedPreconditionError`, naming the pending jobs. Wait
for the restore to finish, then delete.


## Schedule automatic backups

A **backup schedule** takes backups of an index on a recurring cadence. Schedules live on
`pc.backup_schedules` and require a plan that includes backups; without it the server
answers {exc}`~pinecone.errors.exceptions.ForbiddenError` (403), even when the index does
not exist.

```python
schedule = pc.backup_schedules.create(
    index_name="product-search",
    name="daily-compliance",
    frequency="daily",       # "daily" | "weekly" | "monthly"
    retention_days=90,
)
print(schedule.schedule_id)
print(schedule.next_scheduled_run)   # a datetime
```

The server picks the run time and reports it in `next_scheduled_run`; you cannot choose
an hour or timezone. `retention_days` is how long each produced backup is kept; the
schedule model reports it back as `retention_expire_after_days`. Each backup is named
`"{name}-{run timestamp}"`, such as `"daily-compliance-20260403T060000Z"`.

An index can have only **one enabled schedule**. Creating a second fails with
{exc}`~pinecone.errors.exceptions.ConflictError` (409). Pod-based indexes cannot have
schedules.

### List the schedules on an index

Schedules are listed per index, disabled ones included.
{meth}`~pinecone.client.backup_schedules.BackupSchedules.iter_schedules` walks every
page; {meth}`~pinecone.client.backup_schedules.BackupSchedules.list` returns one.

```python
for schedule in pc.backup_schedules.iter_schedules(index_name="product-search"):
    print(schedule.schedule_id, schedule.frequency, schedule.enabled)
```

### Describe, update, and delete a schedule

```python
schedule = pc.backup_schedules.describe(schedule_id="sched-abc123")
```

`update` changes only the arguments you pass:

```python
paused = pc.backup_schedules.update(schedule_id="sched-abc123", enabled=False)
print(paused.next_scheduled_run)   # None while disabled

pc.backup_schedules.update(
    schedule_id="sched-abc123", frequency="weekly", retention_days=30
)
```

```{warning}
Re-enabling a paused schedule with `enabled=True` plans its next run from the time of
the update, not from its old slot, so a pause shifts the cadence. Re-enabling raises
{exc}`~pinecone.errors.exceptions.ConflictError` if another schedule on the index is
already enabled.
```

```python
pc.backup_schedules.delete(schedule_id="sched-abc123")
```

Deleting a schedule stops future runs. Backups it already produced are kept until their
retention period ends. A repeated delete of the same `schedule_id` raises
{exc}`~pinecone.errors.exceptions.NotFoundError`, so when retrying a delete whose
response was lost, treat a 404 as success.

### Inspect what a schedule has produced

```python
for run in pc.backup_schedules.iter_history(schedule_id="sched-abc123"):
    print(run.backup_id, run.status, run.record_count)
```

Each history row is a backup the schedule produced or has planned. The next run is
listed as soon as it is planned, with status `"Initializing"` until it finishes, so filter
on `run.status == "Ready"` for completed backups. Fields such as `record_count` and
`size_bytes` can be `None` on rows that are not finished.
{meth}`~pinecone.client.backup_schedules.BackupSchedules.history` returns one page;
`iter_history` walks all of them.


## See also

- {class}`~pinecone.models.backups.model.BackupModel`: backup response model
- {class}`~pinecone.models.backups.model.RestoreJobModel`: restore job model
- {class}`~pinecone.models.backups.schedules.BackupScheduleModel`: schedule response model
- {class}`~pinecone.models.backups.schedules.BackupScheduleHistoryItem`: one run produced by a schedule
- [Serverless indexes](serverless.md): creating and configuring the indexes you back up
- [Collections](../collections.md): the pod-based snapshot mechanism
