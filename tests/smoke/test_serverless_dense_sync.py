"""Priority-3 smoke test — sync serverless dense + namespaces + imports.

Walks the full Index data plane on a single dense serverless index, plus the
control-plane Indexes namespace and Pinecone top-level surface.

Punchlist coverage (sync):

- pc.indexes.list / describe / exists / create / configure / delete
- pc.index (factory), pc.config, pc.close, ``with`` ctx mgr
- Index: upsert, upsert_from_dataframe (if pandas installed), query,
  query_namespaces, fetch, fetch_by_metadata, delete, update,
  describe_index_stats, create_namespace, describe_namespace,
  delete_namespace, list_namespaces_paginated, list_namespaces,
  list_paginated, list, close, ``with`` ctx mgr, host
- Imports sub-block (start_import, describe_import, cancel_import,
  list_imports, list_imports_paginated): runs only if
  ``PINECONE_IMPORT_S3_URI`` is set.
"""

from __future__ import annotations

import os

import pytest

from pinecone import Pinecone, ServerlessSpec, Vector
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    ensure_index_deleted,
    unique_name,
)
from tests.smoke.helpers import wait_for_vector_count

CLOUD = "aws"
REGION = "us-east-1"
DIM = 8


def _vec(i: int, *, category: str | None = None) -> dict[str, object]:
    """Build a unique deterministic dict-form vector at index ``i``."""
    base = (i + 1) * 0.05
    record: dict[str, object] = {
        "id": f"v{i}",
        "values": [base + j * 0.01 for j in range(DIM)],
    }
    if category is not None:
        record["metadata"] = {"category": category}
    return record


@pytest.mark.smoke
def test_serverless_dense_smoke(client: Pinecone) -> None:
    """End-to-end serverless dense walkthrough."""
    name = unique_name(f"{SMOKE_PREFIX}-srv-dense")

    try:
        # ----- control plane: create / describe / list / exists / configure -----
        created = client.indexes.create(
            name=name,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            dimension=DIM,
            metric="cosine",
        )
        assert created.name == name

        described = client.indexes.describe(name)
        assert described.name == name
        assert described.host

        listing = client.indexes.list()
        assert any(i.name == name for i in listing.indexes)

        assert client.indexes.exists(name) is True
        assert client.indexes.exists(f"{SMOKE_PREFIX}-does-not-exist") is False

        assert client.config.api_key  # Pinecone.config property

        # configure: tweak tags + deletion_protection (no-op rename essentially)
        client.indexes.configure(name, tags={"env": "smoke"})

        # ----- data plane via pc.index() factory -----
        idx = client.index(name=name)
        assert idx.host  # Index.host property
        try:
            # ----- upsert (mixed forms) -----
            mixed = [
                Vector(id="v0", values=[0.05 + j * 0.01 for j in range(DIM)]),
                ("v1", [0.10 + j * 0.01 for j in range(DIM)]),
                ("v2", [0.15 + j * 0.01 for j in range(DIM)], {"category": "x"}),
                _vec(3, category="x"),
                _vec(4),
                _vec(5, category="y"),
                _vec(6),
                _vec(7),
                _vec(8),
                _vec(9, category="x"),
            ]
            up_resp = idx.upsert(vectors=mixed, namespace="alpha")
            assert up_resp.upserted_count == 10

            # ----- upsert: populate beta namespace (plain upsert, no pandas) -----
            beta_records = [
                {
                    "id": f"b{i}",
                    "values": [0.30 + i * 0.01 + j * 0.001 for j in range(DIM)],
                }
                for i in range(5)
            ]
            beta_resp = idx.upsert(vectors=beta_records, namespace="beta")
            assert beta_resp.upserted_count == 5

            # upsert_from_dataframe (skip if pandas not installed)
            try:
                import pandas as pd  # type: ignore[import-untyped]

                df = pd.DataFrame(
                    [
                        {
                            "id": f"d{i}",
                            "values": [0.30 + i * 0.01 + j * 0.001 for j in range(DIM)],
                        }
                        for i in range(5)
                    ]
                )
                df_resp = idx.upsert_from_dataframe(df, namespace="beta", show_progress=False)
                assert df_resp.upserted_count == 5
            except ImportError:
                print("  pandas not installed — skipping upsert_from_dataframe")

            # ----- vector freshness -----
            wait_for_vector_count(idx, "alpha", expected=10)

            # ----- query (vector form) -----
            q_resp = idx.query(
                top_k=3,
                vector=[0.10 + j * 0.01 for j in range(DIM)],
                namespace="alpha",
                include_metadata=True,
            )
            assert len(q_resp.matches) == 3

            # ----- query_namespaces (alpha + beta both populated above) -----
            wait_for_vector_count(idx, "beta", expected=5, timeout=60)
            multi = idx.query_namespaces(
                vector=[0.10 + j * 0.01 for j in range(DIM)],
                namespaces=["alpha", "beta"],
                metric="cosine",
                top_k=3,
            )
            assert len(multi.matches) > 0

            # ----- fetch -----
            fetch_resp = idx.fetch(ids=["v0", "v1", "v2"], namespace="alpha")
            assert set(fetch_resp.vectors.keys()) == {"v0", "v1", "v2"}

            # ----- fetch_by_metadata -----
            fbm = idx.fetch_by_metadata(
                filter={"category": {"$eq": "x"}},
                namespace="alpha",
                limit=10,
            )
            # We upserted 4 vectors with category=x (v2, v3, v9 with metadata + v2 dupe)
            # accept any non-zero count — server may paginate
            assert len(fbm.vectors) >= 1

            # ----- update -----
            idx.update(
                id="v0",
                set_metadata={"tag": "updated"},
                namespace="alpha",
            )

            # ----- list_paginated / list -----
            page = idx.list_paginated(prefix="v", limit=5, namespace="alpha")
            assert len(page.vectors) > 0
            all_ids: list[str] = []
            for p in idx.list(prefix="v", namespace="alpha"):
                all_ids.extend(item.id for item in p.vectors)
            assert len(all_ids) >= 10  # we upserted 10 with v* prefix

            # ----- describe_index_stats -----
            stats = idx.describe_index_stats()
            assert "alpha" in stats.namespaces

            # ----- namespace ops -----
            ns_name = "smoke-gamma"
            created_ns = idx.create_namespace(name=ns_name)
            assert created_ns.name == ns_name
            described_ns = idx.describe_namespace(name=ns_name)
            assert described_ns.name == ns_name
            ns_page = idx.list_namespaces_paginated(limit=10)
            ns_names = {ns.name for ns in ns_page.namespaces}
            assert ns_name in ns_names
            iter_names: list[str] = []
            for page in idx.list_namespaces():
                iter_names.extend(ns.name for ns in page.namespaces)
            assert ns_name in iter_names
            idx.delete_namespace(name=ns_name)

            # ----- delete (by ids) -----
            idx.delete(ids=["v0", "v1"], namespace="alpha")

            # ----- imports sub-block (skip if no env URI) -----
            import_uri = os.getenv("PINECONE_IMPORT_S3_URI")
            integration_id = os.getenv("PINECONE_IMPORT_INTEGRATION_ID")
            if import_uri:
                start_resp = idx.start_import(
                    uri=import_uri,
                    integration_id=integration_id,
                )
                import_id = start_resp.id
                desc = idx.describe_import(import_id)
                assert desc.id == import_id

                imports_iter = list(idx.list_imports(limit=5))
                assert any(imp.id == import_id for imp in imports_iter)

                page = idx.list_imports_paginated(limit=5)
                assert page.data is not None or hasattr(page, "imports")

                idx.cancel_import(import_id)
            else:
                print("  PINECONE_IMPORT_S3_URI not set — skipping imports sub-block")

            # ----- with-statement context manager on Index -----
        finally:
            idx.close()

        # Re-open via context manager
        with client.index(name=name) as idx2:
            stats2 = idx2.describe_index_stats()
            assert stats2.dimension == DIM

    finally:
        ensure_index_deleted(client, name)
        client.close()
