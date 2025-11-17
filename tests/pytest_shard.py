"""
Pytest plugin for sharding tests across multiple CI builds.

This plugin allows splitting the test suite into N shards and running only
the tests in a specified shard. This is useful for parallelizing test runs
across multiple CI jobs.

Usage:
    pytest --splits=3 --group=1  # Run shard 1 of 3
    pytest --splits=3 --group=2  # Run shard 2 of 3
    pytest --splits=3 --group=3  # Run shard 3 of 3

Environment variables:
    PYTEST_SPLITS: Number of shards (alternative to --splits)
    PYTEST_GROUP: Shard number to run (alternative to --group, 1-indexed)
"""

import hashlib
import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for test sharding."""
    group = parser.getgroup("shard", "test sharding options")
    group.addoption(
        "--splits", type=int, default=None, help="Total number of shards to split tests into"
    )
    group.addoption(
        "--group",
        type=int,
        default=None,
        help="Which shard to run (1-indexed, must be between 1 and --splits)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Filter test items based on shard assignment."""
    splits = config.getoption("--splits") or int(os.environ.get("PYTEST_SPLITS", "0"))
    group = config.getoption("--group") or int(os.environ.get("PYTEST_GROUP", "0"))

    # Only activate if splits is provided
    if splits == 0:
        return

    # Validate arguments
    if splits < 1:
        raise pytest.UsageError("--splits must be a positive integer (or set PYTEST_SPLITS)")

    if group < 1:
        raise pytest.UsageError(
            "--group must be a positive integer between 1 and --splits (or set PYTEST_GROUP)"
        )

    if group > splits:
        raise pytest.UsageError(f"--group ({group}) must be between 1 and --splits ({splits})")

    # Assign tests to shards using hash-based distribution
    # This ensures deterministic assignment across runs
    shard_items: list[pytest.Item] = []
    for item in items:
        # Use the test node ID as the basis for hashing
        # nodeid format: "path/to/test_file.py::TestClass::test_method"
        nodeid_bytes = item.nodeid.encode("utf-8")
        hash_value = int(hashlib.md5(nodeid_bytes).hexdigest(), 16)
        # Assign to shard (1-indexed)
        assigned_shard = (hash_value % splits) + 1

        if assigned_shard == group:
            shard_items.append(item)

    # Replace items with only those in the current shard
    original_count = len(items)
    items[:] = shard_items

    # Store shard info for later reporting
    config._shard_info = {
        "group": group,
        "splits": splits,
        "shard_count": len(shard_items),
        "total_count": original_count,
    }
