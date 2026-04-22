# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

```sh
uv sync --extra grpc --extra asyncio   # install all dependencies
uv run pre-commit install               # enable lint/format checks on commit
```

## Key Commands

```sh
make test-unit                          # run unit + grpc unit tests
uv run pytest tests/unit                # REST unit tests only
uv run pytest tests/unit_grpc           # gRPC unit tests only
uv run pytest tests/unit/path/to/test_file.py::ClassName::test_method  # single test

uv run mypy pinecone                    # type-check (excludes pinecone/core/)
uv run ruff check --fix                 # lint
uv run ruff format                      # format

uv run repl                             # interactive REPL with pre-loaded Pinecone client

make generate-oas                       # regenerate pinecone/core/openapi/ from OpenAPI specs
```

Integration tests make live Pinecone API calls and incur cost — only Pinecone employees should run them. Set credentials in `.env` (see `.env.example`) before running.

## Architecture

### Layer Overview

```
Pinecone / PineconeAsyncio   ← public entry point (pinecone/pinecone.py, pinecone_asyncio.py)
    ├── DBControl             ← index/collection/backup management (pinecone/db_control/)
    ├── DBData / Index        ← vector upsert/query/fetch/delete (pinecone/db_data/)
    └── Inference             ← embedding and reranking models (pinecone/inference/)
```

`Pinecone` and `PineconeAsyncio` are thin facades. Each delegates to `DBControl` (control-plane operations) and returns `Index` / `IndexAsyncio` objects (data-plane operations). Inference is accessible via `pc.inference`.

### Generated Code — Never Edit Manually

`pinecone/core/openapi/` is fully generated from OpenAPI specs via `make generate-oas` (which runs `codegen/build-oas.sh`). The script calls the openapi-generator Docker image, applies several post-processing fixes (underscore field name normalization, datetime coercion removal, shared-class deduplication), then runs `ruff format`. **Do not hand-edit files in `pinecone/core/`.**

Shared OpenAPI utilities (ApiClient, exceptions, model_utils, etc.) live in `pinecone/openapi_support/` rather than being duplicated across the five generated modules (`db_control`, `db_data`, `inference`, `oauth`, `admin`).

### Adapter Layer

`pinecone/adapters/` converts generated OpenAPI response objects into clean SDK dataclasses. This isolates the rest of the SDK from generated-model churn. When a new response type is needed, add it here rather than parsing OpenAPI objects in index.py or other business logic files.

### Sync / Async Split

Every stateful class has a sync and an async variant:
- `DBControl` / `DBControlAsyncio`
- `Index` (in `db_data/index.py`) / `IndexAsyncio` (in `db_data/index_asyncio.py`)
- `Inference` / `AsyncioInference`

The async variants use `aiohttp` (optional extra). The sync variants use `urllib3`. gRPC is a third transport option installed via the `grpc` extra; data-plane integration tests can be toggled to gRPC with `USE_GRPC=true`.

### Lazy Imports

`pinecone/__init__.py` defers most imports through `utils/lazy_imports.py` to keep module startup time fast. When adding new public symbols, register them in the lazy import maps in `__init__.py` rather than adding top-level imports. The `.pyi` stub (`__init__.pyi`) is the authoritative type-visible public API surface and must be kept in sync.

### Testing Philosophy

Unit tests are intentionally sparse — they cover data conversion edge cases (e.g. `VectorFactory`, `QueryResultsAggregator`) but not every method. Most confidence comes from integration tests. When writing unit tests, check `tests/unit/db_data/` for patterns. Fixtures and index setup/teardown for integration tests live in `conftest.py` files at each directory level.
