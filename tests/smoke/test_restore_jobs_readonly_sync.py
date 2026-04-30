"""Priority-6 smoke test — sync restore-jobs read-only checks.

Confirms ``pc.restore_jobs.list`` and ``pc.restore_jobs.describe`` succeed
without running an actual restore (which would require a backup, which is
out of scope).

Punchlist coverage (sync): RestoreJobs.list, RestoreJobs.describe.
"""

from __future__ import annotations

import pytest

from pinecone import NotFoundError, Pinecone, PineconeApiException


@pytest.mark.smoke
def test_restore_jobs_readonly_smoke(client: Pinecone) -> None:
    """List restore jobs and describe one (or fall back to a NotFound test)."""
    try:
        result = client.restore_jobs.list(limit=10)
        assert hasattr(result, "data")
        if result.data:
            sample = result.data[0]
            described = client.restore_jobs.describe(job_id=sample.restore_job_id)
            assert described.restore_job_id == sample.restore_job_id
        else:
            with pytest.raises((NotFoundError, PineconeApiException)):
                client.restore_jobs.describe(
                    job_id="smoke-no-such-restore-job-id"
                )
    finally:
        client.close()
