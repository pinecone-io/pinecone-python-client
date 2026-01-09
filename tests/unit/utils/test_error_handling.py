import pytest

from pinecone.utils.error_handling import validate_and_convert_errors, ProtocolError


class TestValidateAndConvertErrors:
    """Test validate_and_convert_errors decorator."""

    def test_successful_function_execution(self):
        """Test that successful function execution is not affected."""

        @validate_and_convert_errors
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_unrelated_exception_passed_through(self):
        """Test that unrelated exceptions are passed through unchanged."""

        @validate_and_convert_errors
        def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            test_func()

    def test_max_retry_error_with_protocol_error_reason(self):
        """Test that MaxRetryError with ProtocolError reason is converted."""

        @validate_and_convert_errors
        def test_func():
            from urllib3.exceptions import MaxRetryError, ProtocolError as Urllib3ProtocolError

            error = MaxRetryError(
                pool=None, url="http://test.com", reason=Urllib3ProtocolError("test")
            )
            raise error

        with pytest.raises(ProtocolError) as exc_info:
            test_func()
        assert "Failed to connect" in str(exc_info.value)
        assert "http://test.com" in str(exc_info.value)

    def test_max_retry_error_with_non_protocol_reason(self):
        """Test that MaxRetryError with non-ProtocolError reason is passed through."""

        from urllib3.exceptions import MaxRetryError

        @validate_and_convert_errors
        def test_func():
            # Create a MaxRetryError with a non-ProtocolError reason (ValueError)
            # We'll use a simple ValueError as the reason
            error = MaxRetryError(pool=None, url="http://test.com", reason=ValueError("test"))
            raise error

        with pytest.raises(MaxRetryError):
            test_func()

    def test_urllib3_protocol_error_converted(self):
        """Test that urllib3 ProtocolError is converted to ProtocolError."""

        @validate_and_convert_errors
        def test_func():
            from urllib3.exceptions import ProtocolError as Urllib3ProtocolError

            raise Urllib3ProtocolError("connection failed")

        with pytest.raises(ProtocolError) as exc_info:
            test_func()
        assert "Connection failed" in str(exc_info.value)
        assert "index host" in str(exc_info.value).lower()

    def test_preserves_function_signature(self):
        """Test that function signature is preserved."""

        @validate_and_convert_errors
        def test_func(arg1: str, arg2: int = 10) -> str:
            return f"{arg1}_{arg2}"

        assert test_func("test", 20) == "test_20"
        assert test_func("test") == "test_10"

    def test_exception_chaining_preserved(self):
        """Test that exception chaining is preserved."""

        @validate_and_convert_errors
        def test_func():
            from urllib3.exceptions import ProtocolError as Urllib3ProtocolError

            original = Urllib3ProtocolError("original error")
            raise original

        with pytest.raises(ProtocolError) as exc_info:
            test_func()
        assert exc_info.value.__cause__ is not None
