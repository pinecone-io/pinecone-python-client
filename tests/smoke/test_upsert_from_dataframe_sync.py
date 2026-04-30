"""Smoke test — sync ``Index.upsert_from_dataframe`` (pandas-gated).

Skipped when pandas is not installed. Run with ``uv run --with pandas``
or by adding pandas to the dev environment.

Punchlist coverage (sync): Index.upsert_from_dataframe.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas", reason="pandas required for upsert_from_dataframe")

from pinecone import Pinecone, ServerlessSpec  # noqa: E402
from pinecone.index import Index  # noqa: E402
from tests.smoke.conftest import (  # noqa: E402
    SMOKE_PREFIX,
    ensure_index_deleted,
    unique_name,
)
from tests.smoke.helpers import wait_for_vector_count  # noqa: E402

CLOUD = "aws"
REGION = "us-east-1"
DIM = 8


@pytest.mark.smoke
def test_upsert_from_dataframe_sync(client: Pinecone) -> None:
    """Upsert a small DataFrame and confirm the count round-trips."""
    name = unique_name(f"{SMOKE_PREFIX}-udf")
    try:
        client.indexes.create(
            name=name,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            dimension=DIM,
            metric="cosine",
        )
        raw_idx = client.index(name=name)
        assert isinstance(raw_idx, Index)
        idx: Index = raw_idx
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
            resp = idx.upsert_from_dataframe(df, namespace="udf", show_progress=False)
            assert resp.upserted_count == 5
            wait_for_vector_count(idx, "udf", expected=5, timeout=60)
        finally:
            idx.close()
    finally:
        ensure_index_deleted(client, name)
        client.close()
