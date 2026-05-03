"""Smoke test — async ``AsyncIndex.upsert_from_dataframe`` (pandas-gated).

Skipped when pandas is not installed. Run with ``uv run --with pandas``
or by adding pandas to the dev environment.

Punchlist coverage (async): AsyncIndex.upsert_from_dataframe.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas", reason="pandas required for upsert_from_dataframe")

from pinecone import AsyncPinecone, ServerlessSpec  # noqa: E402
from tests.smoke.conftest import (  # noqa: E402
    SMOKE_PREFIX,
    async_ensure_index_deleted,
    unique_name,
)
from tests.smoke.helpers import async_wait_for_vector_count  # noqa: E402

CLOUD = "aws"
REGION = "us-east-1"
DIM = 8


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_upsert_from_dataframe_async(api_key: str) -> None:
    """Upsert a small DataFrame via AsyncIndex and confirm the count round-trips."""
    pc = AsyncPinecone(api_key=api_key)
    name = unique_name(f"{SMOKE_PREFIX}-udf-async")
    try:
        await pc.indexes.create(
            name=name,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            dimension=DIM,
            metric="cosine",
        )
        idx = await pc.index(name=name)
        try:
            df = pd.DataFrame(
                [
                    {
                        "id": f"d{i}",
                        "values": [0.30 + i * 0.01 + j * 0.001 for j in range(DIM)],
                    }
                    for i in range(5)
                ]
            )
            resp = await idx.upsert_from_dataframe(df, namespace="udf", show_progress=False)
            assert resp.upserted_count == 5
            await async_wait_for_vector_count(idx, "udf", expected=5, timeout=60)
        finally:
            await idx.close()
    finally:
        await async_ensure_index_deleted(pc, name)
        await pc.close()
