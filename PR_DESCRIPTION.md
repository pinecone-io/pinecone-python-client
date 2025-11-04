# Intelligent CI Test Selection for PRs

## Summary

This PR implements intelligent test selection for pull requests, automatically determining which integration test suites to run based on changed files. This reduces CI time and costs by running only relevant tests while maintaining safety through fallback mechanisms.

## Problem

Previously, all integration test suites ran on every PR regardless of what code changed. This resulted in:
- Unnecessary CI execution time and costs
- Slower feedback cycles for developers
- Resource waste when only a small portion of the codebase changed

## Solution

The implementation analyzes changed files in PRs and maps them to specific test suites. It includes:
- **Automatic test selection**: Runs only test suites relevant to changed code paths
- **Safety fallbacks**: Runs all tests when changes touch critical infrastructure or when analysis fails
- **Manual override**: Option to force running all tests via workflow dispatch

## Changes

### 1. Test Suite Mapping Script (`.github/scripts/determine-test-suites.py`)
- Analyzes git diff to identify changed files
- Maps code paths to test suites:
  - `pinecone/db_control/` → control tests (serverless, resources/index, resources/collections, asyncio variants)
  - `pinecone/db_data/` → data tests (sync, asyncio, gRPC)
  - `pinecone/inference/` → inference tests (sync, asyncio)
  - `pinecone/admin/` → admin tests
  - `pinecone/grpc/` → gRPC-specific tests
  - Plugin-related files → plugin tests
- Identifies critical paths that require full test suite:
  - `pinecone/config/`, `pinecone/core/`, `pinecone/openapi_support/`
  - `pinecone/utils/`, `pinecone/exceptions/`
  - Core interface files (`pinecone.py`, `pinecone_asyncio.py`, etc.)
- Falls back to running all tests if:
  - Script execution fails
  - No files match any mapping
  - Critical paths are touched

### 2. Updated PR Workflow (`.github/workflows/on-pr.yaml`)
- Added `determine-test-suites` job that runs before integration tests
- Added `run_all_tests` input parameter for manual override via workflow dispatch
- Passes selected test suites to integration test workflow
- Includes error handling and validation

### 3. Updated Integration Test Workflow (`.github/workflows/testing-integration.yaml`)
- Added optional inputs for each job type's test suites:
  - `rest_sync_suites_json`
  - `rest_asyncio_suites_json`
  - `grpc_sync_suites_json`
  - `admin_suites_json`
- Filters test matrix based on provided suites
- Skips jobs when their test suite array is empty
- Maintains backward compatibility (runs all tests when inputs not provided)

## Usage

### Automatic (Default)
On every PR, the workflow automatically:
1. Analyzes changed files
2. Determines relevant test suites
3. Runs only those test suites

### Manual Override
To force running all tests on a PR:
1. Go to Actions → "Testing (PR)" workflow
2. Click "Run workflow"
3. Check "Run all integration tests regardless of changes"
4. Run the workflow

## Safety Features

1. **Critical path detection**: Changes to core infrastructure (config, utils, exceptions, etc.) trigger full test suite
2. **Fallback on failure**: If the analysis script fails, falls back to running all tests
3. **Empty result handling**: If no tests match, runs all tests as a safety measure
4. **Main branch unchanged**: Main branch workflows continue to run all tests

## Example Scenarios

### Scenario 1: Change only `pinecone/db_data/index.py`
- **Runs**: `data`, `data_asyncio`, `data_grpc_futures` test suites
- **Skips**: `control/*`, `inference/*`, `admin`, `plugins` test suites
- **Result**: ~70% reduction in test execution

### Scenario 2: Change `pinecone/config/pinecone_config.py`
- **Runs**: All test suites (critical path)
- **Reason**: Configuration changes affect all functionality

### Scenario 3: Change `pinecone/inference/inference.py`
- **Runs**: `inference/sync`, `inference/asyncio` test suites
- **Skips**: Other test suites
- **Result**: ~85% reduction in test execution

## Testing

The implementation has been tested with:
- ✅ YAML syntax validation
- ✅ Python script syntax validation
- ✅ Test suite mapping logic verification
- ✅ Edge case handling (empty arrays, failures, etc.)

## Benefits

- **Cost savings**: Reduce CI costs by running only relevant tests
- **Faster feedback**: Developers get test results faster when only subset runs
- **Better resource utilization**: CI runners are used more efficiently
- **Maintainability**: Easy to update mappings as codebase evolves

## Backward Compatibility

- Main branch workflows unchanged (still run all tests)
- PR workflows backward compatible (can manually trigger full suite)
- Existing test suite structure unchanged
- No changes to test code itself

## Future Improvements

Potential enhancements for future PRs:
- Track test execution time savings
- Add metrics/logging for test selection decisions
- Fine-tune mappings based on actual usage patterns
- Consider test dependencies (e.g., if A changes, also run B)
