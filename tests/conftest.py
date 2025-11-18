"""
Root-level conftest.py for the tests directory.

This file registers pytest plugins that should be available for all tests.
"""

# Register pytest shard plugin globally
pytest_plugins = ["tests.pytest_shard"]
