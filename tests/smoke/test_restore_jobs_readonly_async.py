"""Priority-6 smoke test — async restore-jobs read-only checks.

Mirror of ``test_restore_jobs_readonly_sync.py``.
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone, NotFoundError, PineconeApiException


@pytest.mark.smoke
@pytest.mark.anyio
async def test_restore_jobs_readonly_smoke_async(async_client: AsyncPinecone) -> None:
    result = await async_client.restore_jobs.list(limit=10)
    assert hasattr(result, "data")
    if result.data:
        sample = result.data[0]
        described = await async_client.restore_jobs.describe(job_id=sample.restore_job_id)
        assert described.restore_job_id == sample.restore_job_id
    else:
        with pytest.raises((NotFoundError, PineconeApiException)):
            await async_client.restore_jobs.describe(job_id="smoke-no-such-restore-job-id-async")
