"""Smoke test — gRPC ``GrpcIndex.upsert_from_dataframe`` (pandas-gated).

Skipped when pandas is not installed. Run with ``uv run --with pandas``
or by adding pandas to the dev environment.

Punchlist coverage (gRPC): GrpcIndex.upsert_from_dataframe.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas", reason="pandas required for upsert_from_dataframe")

from pinecone import GrpcIndex, Pinecone, ServerlessSpec  # noqa: E402
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
def test_upsert_from_dataframe_grpc(client: Pinecone) -> None:
    """Upsert a small DataFrame via GrpcIndex and confirm the count round-trips."""
    name = unique_name(f"{SMOKE_PREFIX}-udf-grpc")
    try:
        client.indexes.create(
            name=name,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            dimension=DIM,
            metric="cosine",
        )
        idx = client.index(name=name, grpc=True)
        assert isinstance(idx, GrpcIndex)
        try:
            df = pd.DataFrame(
                [
                    {
                        "id": f"d{i}",
                        "values": [0.50 + 0.01 * i + j * 0.001 for j in range(DIM)],
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
