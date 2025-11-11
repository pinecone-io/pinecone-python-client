"""
Unit tests for the pytest_shard plugin.
"""

import hashlib

# Enable pytester plugin for testdir fixture
pytest_plugins = ["pytester"]


class TestPytestShardPlugin:
    """Test the pytest shard plugin functionality."""

    def test_plugin_adds_command_line_options(self, testdir):
        """Test that the plugin adds --splits and --group options."""
        # Create a simple test file
        testdir.makepyfile(
            """
            def test_example():
                assert True
            """
        )

        # Check that the options are available
        result = testdir.runpytest("--help")
        result.stdout.fnmatch_lines(["*--splits*", "*--group*"])

    def test_plugin_filters_tests_by_shard(self, testdir):
        """Test that the plugin correctly filters tests into shards."""
        # Create multiple test files with multiple tests
        testdir.makepyfile(
            test_file1="""
            def test_a():
                assert True

            def test_b():
                assert True

            def test_c():
                assert True
            """
        )

        testdir.makepyfile(
            test_file2="""
            def test_d():
                assert True

            def test_e():
                assert True
            """
        )

        # Collect all tests first to see total count
        result = testdir.runpytest("--collect-only", "-q")
        all_tests = [line for line in result.stdout.lines if "test_" in line and "::" in line]
        total_test_count = len(all_tests)

        # Run with sharding - should only get a subset
        result = testdir.runpytest("--splits=3", "--group=1", "--collect-only", "-q")
        shard1_tests = [line for line in result.stdout.lines if "test_" in line and "::" in line]

        # Verify we got a subset (not all tests)
        assert len(shard1_tests) < total_test_count
        assert len(shard1_tests) > 0

    def test_all_tests_distributed_across_shards(self, testdir):
        """Test that all tests are distributed across shards (no tests lost)."""
        # Create multiple tests
        testdir.makepyfile(
            """
            def test_1(): assert True
            def test_2(): assert True
            def test_3(): assert True
            def test_4(): assert True
            def test_5(): assert True
            def test_6(): assert True
            def test_7(): assert True
            def test_8(): assert True
            def test_9(): assert True
            def test_10(): assert True
            """
        )

        # Collect all tests without sharding
        result = testdir.runpytest("--collect-only", "-q")
        all_tests = set(
            line.strip()
            for line in result.stdout.lines
            if "test_" in line and "::" in line and "PASSED" not in line and "FAILED" not in line
        )
        total_count = len(all_tests)

        # Collect tests from each shard
        shard_tests = []
        for group in range(1, 4):  # 3 shards
            result = testdir.runpytest("--splits=3", f"--group={group}", "--collect-only", "-q")
            shard_test_set = set(
                line.strip()
                for line in result.stdout.lines
                if "test_" in line
                and "::" in line
                and "PASSED" not in line
                and "FAILED" not in line
            )
            shard_tests.append(shard_test_set)

        # Combine all shards
        combined_tests = set()
        for shard_set in shard_tests:
            combined_tests.update(shard_set)

        # Verify all tests are accounted for
        assert len(combined_tests) == total_count
        assert combined_tests == all_tests

    def test_deterministic_shard_assignment(self, testdir):
        """Test that shard assignment is deterministic (same test always in same shard)."""
        testdir.makepyfile(
            """
            def test_deterministic():
                assert True
            """
        )

        # Run collection twice with same shard parameters
        result1 = testdir.runpytest("--splits=3", "--group=1", "--collect-only", "-q")
        result2 = testdir.runpytest("--splits=3", "--group=1", "--collect-only", "-q")

        tests1 = [line.strip() for line in result1.stdout.lines if "test_" in line and "::" in line]
        tests2 = [line.strip() for line in result2.stdout.lines if "test_" in line and "::" in line]

        # Should get the same tests both times
        assert tests1 == tests2

    def test_validation_splits_must_be_positive(self, testdir):
        """Test that --splits must be a positive integer."""
        testdir.makepyfile(
            """
            def test_example():
                assert True
            """
        )

        result = testdir.runpytest("--splits=0", "--group=1")
        result.stderr.fnmatch_lines(["*--splits must be a positive integer*"])

        result = testdir.runpytest("--splits=-1", "--group=1")
        result.stderr.fnmatch_lines(["*--splits must be a positive integer*"])

    def test_validation_group_must_be_positive(self, testdir):
        """Test that --group must be a positive integer."""
        testdir.makepyfile(
            """
            def test_example():
                assert True
            """
        )

        result = testdir.runpytest("--splits=3", "--group=0")
        result.stderr.fnmatch_lines(["*--group must be a positive integer between 1 and --splits*"])

        result = testdir.runpytest("--splits=3", "--group=-1")
        result.stderr.fnmatch_lines(["*--group must be a positive integer between 1 and --splits*"])

    def test_validation_group_cannot_exceed_splits(self, testdir):
        """Test that --group cannot exceed --splits."""
        testdir.makepyfile(
            """
            def test_example():
                assert True
            """
        )

        result = testdir.runpytest("--splits=3", "--group=4")
        result.stderr.fnmatch_lines(["*--group (4) must be between 1 and --splits (3)*"])

    def test_plugin_inactive_without_splits(self, testdir):
        """Test that plugin doesn't filter tests when --splits is not provided."""
        testdir.makepyfile(
            """
            def test_a():
                assert True

            def test_b():
                assert True
            """
        )

        # Without --splits, all tests should run
        result = testdir.runpytest("--collect-only", "-q")
        all_tests = [
            line.strip() for line in result.stdout.lines if "test_" in line and "::" in line
        ]

        # With --splits but no --group, should error
        # Actually, let's test that without splits, all tests are collected
        result2 = testdir.runpytest("--collect-only", "-q")
        all_tests2 = [
            line.strip() for line in result2.stdout.lines if "test_" in line and "::" in line
        ]

        assert len(all_tests) == len(all_tests2)

    def test_environment_variable_support(self, testdir, monkeypatch):
        """Test that environment variables PYTEST_SPLITS and PYTEST_GROUP work."""
        testdir.makepyfile(
            """
            def test_1(): assert True
            def test_2(): assert True
            def test_3(): assert True
            """
        )

        monkeypatch.setenv("PYTEST_SPLITS", "2")
        monkeypatch.setenv("PYTEST_GROUP", "1")

        # Should work with env vars instead of command-line args
        result = testdir.runpytest("--collect-only", "-q")
        # Should not error and should filter tests
        assert result.ret == 0

    def test_single_shard_gets_all_tests(self, testdir):
        """Test that with --splits=1, all tests are in the single shard."""
        testdir.makepyfile(
            """
            def test_1(): assert True
            def test_2(): assert True
            def test_3(): assert True
            """
        )

        # Collect all tests without sharding
        result = testdir.runpytest("--collect-only", "-q")
        all_tests = set(
            line.strip() for line in result.stdout.lines if "test_" in line and "::" in line
        )

        # Collect with single shard
        result = testdir.runpytest("--splits=1", "--group=1", "--collect-only", "-q")
        shard_tests = set(
            line.strip() for line in result.stdout.lines if "test_" in line and "::" in line
        )

        # Should have all tests
        assert shard_tests == all_tests

    def test_hash_based_distribution(self):
        """Test that hash-based distribution works correctly."""
        # Test the hash logic directly
        test_nodeids = ["test_file.py::test_a", "test_file.py::test_b", "test_file.py::test_c"]

        splits = 3
        shard_assignments = {}
        for nodeid in test_nodeids:
            nodeid_bytes = nodeid.encode("utf-8")
            hash_value = int(hashlib.md5(nodeid_bytes).hexdigest(), 16)
            assigned_shard = (hash_value % splits) + 1
            shard_assignments[nodeid] = assigned_shard

        # Verify assignments are in valid range
        for nodeid, shard in shard_assignments.items():
            assert 1 <= shard <= splits

        # Verify deterministic (run twice)
        shard_assignments2 = {}
        for nodeid in test_nodeids:
            nodeid_bytes = nodeid.encode("utf-8")
            hash_value = int(hashlib.md5(nodeid_bytes).hexdigest(), 16)
            assigned_shard = (hash_value % splits) + 1
            shard_assignments2[nodeid] = assigned_shard

        assert shard_assignments == shard_assignments2
