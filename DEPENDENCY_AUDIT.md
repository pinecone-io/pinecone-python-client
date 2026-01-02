# Dependency Audit Report

Generated: 2025-01-XX

## Summary

This audit identifies dependencies that could be removed or upgraded in the Pinecone Python client.

## Dependencies That Could Be Removed

### 1. `lz4` (grpc optional dependency)
**Status**: ⚠️ **POTENTIALLY UNUSED**

- **Location**: `pyproject.toml` line 47
- **Current version**: `>=3.1.3`
- **Usage**: No imports or usage found in the codebase
- **Note**: gRPC has built-in compression support (Gzip, Deflate) via `grpc.Compression` enum, which is used in `pinecone/grpc/grpc_runner.py`. The `lz4` package doesn't appear to be necessary for gRPC compression.
- **Recommendation**: Remove unless there's a specific use case for LZ4 compression that isn't visible in the codebase.

### 2. `beautifulsoup4` (dev optional dependency)
**Status**: ⚠️ **UNUSED**

- **Location**: `pyproject.toml` line 79
- **Current version**: `>=4.13.3,<5.0.0`
- **Usage**: No imports or usage found in the codebase
- **Recommendation**: Remove unless it's used in documentation generation scripts that aren't tracked in the repo.

### 3. `responses` (dev optional dependency)
**Status**: ⚠️ **UNUSED**

- **Location**: `pyproject.toml` line 77
- **Current version**: `>=0.8.1`
- **Usage**: No imports found in test files
- **Recommendation**: Remove if not used for mocking HTTP responses in tests.

### 4. `urllib3_mock` (dev optional dependency)
**Status**: ⚠️ **UNUSED**

- **Location**: `pyproject.toml` line 76
- **Current version**: `==0.3.3`
- **Usage**: No imports found in test files
- **Recommendation**: Remove if not used for mocking urllib3 in tests.

### 5. `protoc-gen-openapiv2` (grpc optional dependency)
**Status**: ⚠️ **POTENTIALLY UNUSED**

- **Location**: `pyproject.toml` line 49
- **Current version**: `>=0.0.1,<0.1.0`
- **Usage**: Only referenced in GitHub workflows for dependency testing, not in actual build scripts
- **Note**: This is a protobuf code generation plugin, but it's not used in `codegen/build-grpc.sh` or `codegen/build-oas.sh`
- **Recommendation**: Verify if this is needed for code generation. If not, remove.

## Dependencies That Are Used (Keep These)

### Development Tools
- `vprof` and `tuna`: Used for profiling/debugging (documented in `docs/maintainers/testing-guide.md`)
- `myst-parser`: Used in `docs/conf.py` for Sphinx documentation with Markdown support
- `pytest*`: Used extensively in tests
- `ruff`: Used for linting and formatting
- `sphinx`: Used for documentation generation
- `python-dotenv`: Used in test configuration files

### Runtime Dependencies
- All core dependencies (`typing-extensions`, `certifi`, `orjson`, `python-dateutil`, `urllib3`) are used
- `aiohttp` and `aiohttp-retry`: Used for async REST API calls
- `grpcio`, `protobuf`, `googleapis-common-protos`: Used for gRPC functionality

## Dependencies That Could Be Upgraded

### Core Dependencies
1. **`typing-extensions`**: Currently `>=3.7.4`
   - Latest: 4.15.0 (already installed per `uv tree`)
   - Status: ✅ Already at latest

2. **`certifi`**: Currently `>=2019.11.17`
   - Latest: 2025.11.12 (already installed per `uv tree`)
   - Status: ✅ Already at latest

3. **`orjson`**: Currently `>=3.0.0`
   - Latest: 3.11.4 (already installed per `uv tree`)
   - Status: ✅ Already at latest

4. **`python-dateutil`**: Currently `>=2.5.3`
   - Latest: 2.9.0.post0 (already installed per `uv tree`)
   - Status: ✅ Already at latest

5. **`urllib3`**: Currently `>=1.26.0` (Python <3.12), `>=1.26.5` (Python >=3.12)
   - Latest: 2.5.0 (already installed per `uv tree`)
   - Status: ✅ Already at latest

### Development Dependencies
1. **`pytest`**: Currently `==8.2.0` (pinned)
   - Latest: 8.3.x available
   - Status: ⚠️ Consider updating to latest 8.x version

2. **`pytest-cov`**: Currently `==2.10.1` (pinned)
   - Latest: 6.x available
   - Status: ⚠️ Significantly outdated, consider upgrading

3. **`pytest-mock`**: Currently `==3.6.1` (pinned)
   - Latest: 3.14.x available
   - Status: ⚠️ Consider updating to latest 3.x version

4. **`pytest-timeout`**: Currently `==2.2.0` (pinned)
   - Latest: 2.3.x available
   - Status: ⚠️ Consider updating to latest 2.x version

5. **`ruff`**: Currently `>=0.9.3,<0.10.0`
   - Latest: 0.9.10 (already installed per `uv tree`)
   - Status: ✅ Already at latest in range

6. **`sphinx`**: Currently `>=7.4.7,<8.0.0` (Python <3.11), `>=8.2.3,<9.0.0` (Python >=3.11)
   - Status: ✅ Version constraints are appropriate

## Recommendations

### High Priority
1. **Remove unused dependencies**: `beautifulsoup4`, `responses`, `urllib3_mock`
2. **Investigate and potentially remove**: `lz4`, `protoc-gen-openapiv2`

### Medium Priority
1. **Upgrade pinned test dependencies**: Consider updating `pytest-cov` from 2.10.1 to latest 6.x (may require test updates)
2. **Update other pinned dependencies**: Update `pytest`, `pytest-mock`, `pytest-timeout` to latest patch versions

### Low Priority
1. **Review dependency version constraints**: Some dependencies use very old minimum versions (e.g., `certifi>=2019.11.17`) which could be updated, though this is low priority as they're already resolving to latest versions.

## Action Items

1. [ ] Remove `beautifulsoup4` from dev dependencies
2. [ ] Remove `responses` from dev dependencies (verify no usage first)
3. [ ] Remove `urllib3_mock` from dev dependencies (verify no usage first)
4. [ ] Investigate `lz4` usage - remove if not needed for gRPC
5. [ ] Investigate `protoc-gen-openapiv2` usage - remove if not needed for code generation
6. [ ] Update `pytest-cov` to latest 6.x version (test compatibility)
7. [ ] Update other pinned pytest dependencies to latest patch versions

## Notes

- All core runtime dependencies are already resolving to their latest versions
- The `uv tree` output shows dependencies are generally up-to-date
- Some dependencies are pinned for stability (pytest ecosystem)
- Optional dependencies (grpc, asyncio, types, dev) are correctly separated
