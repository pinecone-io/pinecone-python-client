#!/usr/bin/env python3
"""
Determine which integration test suites to run based on changed files in a PR.

This script analyzes git diff to identify changed files and maps them to test suites.
Critical paths trigger running all tests for safety.
"""

import json
import subprocess
import sys
from typing import Set


# Define all possible test suites organized by job type
ALL_REST_SYNC_SUITES = [
    "control/serverless",
    "control/resources/index",
    "control/resources/collections",
    "inference/sync",
    "plugins",
    "data",
]

ALL_REST_ASYNCIO_SUITES = [
    "control_asyncio/resources/index",
    "control_asyncio/*.py",
    "inference/asyncio",
    "data_asyncio",
]

ALL_GRPC_SYNC_SUITES = ["data", "data_grpc_futures"]

ALL_ADMIN_SUITES = ["admin"]

# Critical paths that require running all tests
CRITICAL_PATHS = [
    "pinecone/config/",
    "pinecone/core/",
    "pinecone/openapi_support/",
    "pinecone/utils/",
    "pinecone/exceptions/",  # Used across all test suites for error handling
    "pinecone/pinecone.py",
    "pinecone/pinecone_asyncio.py",
    "pinecone/pinecone_interface_asyncio.py",  # Core asyncio interface
    "pinecone/legacy_pinecone_interface.py",  # Legacy interface affects many tests
    "pinecone/deprecation_warnings.py",  # Affects all code paths
    "pinecone/__init__.py",
    "pinecone/__init__.pyi",
]

# Path to test suite mappings
# Format: (path_pattern, [list of test suites])
PATH_MAPPINGS = [
    # db_control mappings
    (
        "pinecone/db_control/",
        [
            "control/serverless",
            "control/resources/index",
            "control/resources/collections",
            "control_asyncio/resources/index",
            "control_asyncio/*.py",
        ],
    ),
    # db_data mappings
    ("pinecone/db_data/", ["data", "data_asyncio", "data_grpc_futures"]),
    # inference mappings
    ("pinecone/inference/", ["inference/sync", "inference/asyncio"]),
    # admin mappings
    ("pinecone/admin/", ["admin"]),
    # grpc mappings
    (
        "pinecone/grpc/",
        [
            "data_grpc_futures",
            "data",  # grpc affects data tests too
        ],
    ),
    # plugin mappings
    ("pinecone/deprecated_plugins.py", ["plugins"]),
    ("pinecone/langchain_import_warnings.py", ["plugins"]),
]


def get_changed_files(base_ref: str = "main") -> Set[str]:
    """Get list of changed files compared to base branch."""
    try:
        # For PRs, compare against the base branch
        # For local testing, compare against HEAD
        result = subprocess.run(
            ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = {line.strip() for line in result.stdout.strip().split("\n") if line.strip()}
        return files
    except subprocess.CalledProcessError:
        # Fallback: try comparing against HEAD~1 for local testing
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1"], capture_output=True, text=True, check=True
            )
            files = {line.strip() for line in result.stdout.strip().split("\n") if line.strip()}
            return files
        except subprocess.CalledProcessError:
            # If git commands fail, return empty set (will trigger full suite)
            return set()


def is_critical_path(file_path: str) -> bool:
    """Check if a file path is in a critical area that requires all tests."""
    return any(file_path.startswith(critical) for critical in CRITICAL_PATHS)


def map_file_to_test_suites(file_path: str) -> Set[str]:
    """Map a single file path to its relevant test suites."""
    suites = set()

    for path_pattern, test_suites in PATH_MAPPINGS:
        if file_path.startswith(path_pattern):
            suites.update(test_suites)

    return suites


def determine_test_suites(changed_files: Set[str], run_all: bool = False) -> dict:
    """
    Determine which test suites to run based on changed files.

    Returns a dict with keys: rest_sync, rest_asyncio, grpc_sync, admin
    Each value is a list of test suite names to run.
    """
    if run_all or not changed_files:
        # Run all tests if explicitly requested or no files changed
        return {
            "rest_sync": ALL_REST_SYNC_SUITES,
            "rest_asyncio": ALL_REST_ASYNCIO_SUITES,
            "grpc_sync": ALL_GRPC_SYNC_SUITES,
            "admin": ALL_ADMIN_SUITES,
        }

    # Check for critical paths
    has_critical = any(is_critical_path(f) for f in changed_files)
    if has_critical:
        # Run all tests if critical paths are touched
        return {
            "rest_sync": ALL_REST_SYNC_SUITES,
            "rest_asyncio": ALL_REST_ASYNCIO_SUITES,
            "grpc_sync": ALL_GRPC_SYNC_SUITES,
            "admin": ALL_ADMIN_SUITES,
        }

    # Map files to test suites
    rest_sync_suites = set()
    rest_asyncio_suites = set()
    grpc_sync_suites = set()
    admin_suites = set()

    for file_path in changed_files:
        # Skip non-Python files and test files
        if not file_path.startswith("pinecone/"):
            continue

        suites = map_file_to_test_suites(file_path)

        # Categorize suites by job type
        for suite in suites:
            if suite in ALL_REST_SYNC_SUITES:
                rest_sync_suites.add(suite)
            if suite in ALL_REST_ASYNCIO_SUITES:
                rest_asyncio_suites.add(suite)
            if suite in ALL_GRPC_SYNC_SUITES:
                grpc_sync_suites.add(suite)
            if suite in ALL_ADMIN_SUITES:
                admin_suites.add(suite)

    # If no tests matched, run all (safety fallback)
    if not (rest_sync_suites or rest_asyncio_suites or grpc_sync_suites or admin_suites):
        return {
            "rest_sync": ALL_REST_SYNC_SUITES,
            "rest_asyncio": ALL_REST_ASYNCIO_SUITES,
            "grpc_sync": ALL_GRPC_SYNC_SUITES,
            "admin": ALL_ADMIN_SUITES,
        }

    return {
        "rest_sync": sorted(list(rest_sync_suites)),
        "rest_asyncio": sorted(list(rest_asyncio_suites)),
        "grpc_sync": sorted(list(grpc_sync_suites)),
        "admin": sorted(list(admin_suites)),
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Determine test suites to run based on changed files"
    )
    parser.add_argument(
        "--base-ref", default="main", help="Base branch/ref to compare against (default: main)"
    )
    parser.add_argument("--run-all", action="store_true", help="Force running all test suites")
    parser.add_argument(
        "--output-format",
        choices=["json", "json-pretty"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    changed_files = get_changed_files(args.base_ref)
    test_suites = determine_test_suites(changed_files, run_all=args.run_all)

    # Output as JSON
    if args.output_format == "json-pretty":
        print(json.dumps(test_suites, indent=2))
    else:
        print(json.dumps(test_suites))

    # Exit with non-zero if no test suites selected (shouldn't happen with safety fallback)
    if not any(test_suites.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
