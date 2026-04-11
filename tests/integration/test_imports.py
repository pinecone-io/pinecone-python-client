"""Integration tests for bulk import lifecycle (sync REST).

Phase 4 area tag: import-lifecycle
Transport: rest (sync), grpc: N/A

Tests Index.start_import(), describe_import(), list_imports(),
list_imports_paginated(), and cancel_import().  Uses a public S3 URI that
the API accepts and queues; the import is cancelled before it can run so
no actual data is transferred.
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse
from tests.integration.conftest import cleanup_resource, unique_name

# A public S3 URI that the Pinecone import API accepts. The import will
# be cancelled before it runs, so it does not matter whether the bucket
# contains Parquet files.
_TEST_URI = "s3://pinecone-test-public/"


# ---------------------------------------------------------------------------
# import-lifecycle — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_import_lifecycle_rest(client: Pinecone) -> None:
    """Full import lifecycle via sync REST: create a serverless index, call
    start_import(), describe_import(), list_imports(), list_imports_paginated(),
    and cancel_import().  Verifies ImportModel and StartImportResponse structures.

    Area tag: import-lifecycle
    Transport: rest
    """
    index_name = unique_name("idx")
    import_id: str | None = None

    try:
        # 1. Create a serverless index (needed to get a data-plane host)
        client.indexes.create(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=120,
        )

        # 2. Get a data-plane Index handle
        idx = client.index(name=index_name)

        # 3. start_import — verify StartImportResponse structure
        start_resp = idx.start_import(uri=_TEST_URI, error_mode="continue")
        assert isinstance(start_resp, StartImportResponse)
        assert isinstance(start_resp.id, str)
        assert start_resp.id != ""
        import_id = start_resp.id

        # 4. describe_import — verify ImportModel fields
        import_op = idx.describe_import(import_id)
        assert isinstance(import_op, ImportModel)
        assert import_op.id == import_id
        assert import_op.status in ("Pending", "InProgress", "Failed", "Completed", "Cancelled")
        assert isinstance(import_op.uri, str)
        assert import_op.uri != ""
        assert isinstance(import_op.created_at, str)
        assert import_op.created_at != ""
        # Optional numeric fields may be None or a number
        assert import_op.percent_complete is None or isinstance(import_op.percent_complete, float)
        assert import_op.records_imported is None or isinstance(import_op.records_imported, int)

        # 5. list_imports — import appears in the result iterator
        all_imports = list(idx.list_imports())
        ids_in_list = [imp.id for imp in all_imports]
        assert import_id in ids_in_list, (
            f"Import {import_id!r} not found in list_imports() result (got: {ids_in_list})"
        )
        for imp in all_imports:
            assert isinstance(imp, ImportModel)
            assert isinstance(imp.id, str)
            assert isinstance(imp.status, str)

        # 6. list_imports_paginated — returns an ImportList for a single page
        page = idx.list_imports_paginated(limit=10)
        assert isinstance(page, ImportList)
        assert hasattr(page, "__iter__")
        assert hasattr(page, "__len__")
        for imp in page:
            assert isinstance(imp, ImportModel)

        # 7. cancel_import — completes without error (returns None)
        result = idx.cancel_import(import_id)
        assert result is None

        # Verify cancelled status
        cancelled_op = idx.describe_import(import_id)
        assert cancelled_op.status == "Cancelled"
        import_id = None  # Successfully cancelled; skip cleanup

    finally:
        if import_id is not None:
            # Best-effort cancel if the test failed before reaching step 7
            try:
                idx.cancel_import(import_id)
                print(f"  Cleaned up import: {import_id}")
            except Exception as exc:
                print(f"  WARNING: Failed to cancel import {import_id}: {exc}")
        cleanup_resource(
            lambda: client.indexes.delete(index_name),
            index_name,
            "index",
        )
