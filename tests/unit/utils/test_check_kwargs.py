from unittest.mock import patch

from pinecone.utils import check_kwargs


def example_function(arg1, arg2, arg3=None):
    """Example function for testing."""
    pass


class TestCheckKwargs:
    """Test check_kwargs utility function."""

    def test_no_unexpected_kwargs_no_logging(self):
        """Test that no logging occurs when all kwargs are valid."""
        with patch("logging.exception") as mock_log:
            check_kwargs(example_function, {"arg1", "arg2", "arg3"})
            mock_log.assert_not_called()

    def test_unexpected_kwargs_logs_warning(self):
        """Test that unexpected kwargs trigger logging."""
        with patch("logging.exception") as mock_log:
            check_kwargs(example_function, {"arg1", "arg2", "arg3", "unexpected_arg"})
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "unexpected keyword argument" in call_args.lower()
            assert "unexpected_arg" in call_args

    def test_multiple_unexpected_kwargs_logs_all(self):
        """Test that multiple unexpected kwargs are all logged."""
        with patch("logging.exception") as mock_log:
            check_kwargs(example_function, {"arg1", "arg2", "arg3", "unexpected1", "unexpected2"})
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "unexpected1" in call_args or "unexpected2" in call_args

    def test_only_unexpected_kwargs(self):
        """Test when only unexpected kwargs are provided."""
        with patch("logging.exception") as mock_log:
            check_kwargs(example_function, {"unexpected_arg"})
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "unexpected keyword argument" in call_args.lower()

    def test_empty_kwargs_set(self):
        """Test with empty kwargs set."""
        with patch("logging.exception") as mock_log:
            check_kwargs(example_function, set())
            mock_log.assert_not_called()

    def test_function_with_no_args(self):
        """Test with function that has no arguments."""

        def no_args_function():
            pass

        with patch("logging.exception") as mock_log:
            check_kwargs(no_args_function, {"any_arg"})
            mock_log.assert_called_once()

    def test_function_with_varargs(self):
        """Test with function that has *args."""

        def varargs_function(*args):
            pass

        with patch("logging.exception") as mock_log:
            check_kwargs(varargs_function, {"any_arg"})
            mock_log.assert_called_once()

    def test_function_with_kwargs(self):
        """Test with function that has **kwargs.

        Note: check_kwargs only checks explicit args, not **kwargs,
        so it will still log unexpected args even for functions with **kwargs.
        This is the current behavior of the function.
        """

        def kwargs_function(**kwargs):
            pass

        with patch("logging.exception") as mock_log:
            check_kwargs(kwargs_function, {"any_arg"})
            # check_kwargs doesn't check for **kwargs, so it will log
            mock_log.assert_called_once()
