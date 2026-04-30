# Smoke test environment variables

The smoke suite (`tests/smoke/`) is a notebook-style end-to-end
walkthrough that hits a real Pinecone backend. Each scenario lives in
its own file and exercises one slice of the public surface. The
following environment variables control which scenarios run.

## Required

| Variable | Effect |
|---|---|
| `PINECONE_API_KEY` | API key for the project under test. The whole suite is skipped if unset. Read from `.env` at the SDK root. |

## Optional (opt-in coverage)

Some scenarios are slow or require additional backend configuration
and are skipped by default. Set the relevant variable to opt in.

| Variable | Effect | Cost / requirement |
|---|---|---|
| `SMOKE_RUN_POD_SHIMS=1` | Enables the pod-index + collection sub-block inside `test_deprecated_shims_{sync,async}.py`. | Pod indexes take ~10-15 minutes to provision and tear down. |
| `SMOKE_RUN_BACKUPS=1` | Enables `test_backups_{sync,async}.py`. | Project must have a configured data integration; backups take seconds-to-minutes server-side. |
| `PINECONE_IMPORT_S3_URI` | Enables the imports sub-block (`start_import` / `describe_import` / `list_imports` / `list_imports_paginated` / `cancel_import`) inside `test_serverless_dense_{sync,async}.py`. Must be a parquet S3 or GCS URI the project's storage integration can read. | Project must have a configured storage integration. Often paired with `PINECONE_IMPORT_INTEGRATION_ID`. |
| `PINECONE_IMPORT_INTEGRATION_ID` | Optional integration ID passed to `start_import`. Pair with `PINECONE_IMPORT_S3_URI`. | Project-specific. |

## Running the suite

```bash
cd sdks/python-sdk2

# Fast subset (priorities 1+2)
uv run --with python-dotenv --with pytest-asyncio pytest \
  tests/smoke/test_inference_sync.py \
  tests/smoke/test_inference_async.py \
  tests/smoke/test_deprecated_shims_sync.py \
  tests/smoke/test_deprecated_shims_async.py -v -s

# Full suite (excludes opt-in scenarios)
uv run --with python-dotenv --with pytest-asyncio pytest tests/smoke/ -v -s

# Full + every opt-in path
SMOKE_RUN_POD_SHIMS=1 SMOKE_RUN_BACKUPS=1 \
  PINECONE_IMPORT_S3_URI=s3://my-bucket/data.parquet \
  PINECONE_IMPORT_INTEGRATION_ID=int-abc123 \
  uv run --with python-dotenv --with pytest-asyncio pytest tests/smoke/ -v -s

# Orphan cleanup
uv run --with python-dotenv python tests/smoke/scripts/cleanup_orphans.py --dry-run
uv run --with python-dotenv python tests/smoke/scripts/cleanup_orphans.py
```

## Adding a new env-var-gated scenario

Use the module-level skip pattern so the whole test file is reported
as `SKIPPED` (not partially-passed) when the gate is unset:

```python
import os
import pytest

if os.getenv("SMOKE_RUN_NEW_THING") != "1":
    pytest.skip(
        "SMOKE_RUN_NEW_THING=1 not set — opt in to run.",
        allow_module_level=True,
    )
```

Then add the variable to the table above so future contributors can
discover it.
