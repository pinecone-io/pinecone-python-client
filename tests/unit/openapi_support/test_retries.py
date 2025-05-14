import pytest
from unittest.mock import patch, MagicMock
from urllib3.exceptions import MaxRetryError
from pinecone.openapi_support.retries import JitterRetry


def test_jitter_retry_backoff():
    """Test that the backoff time includes jitter."""
    retry = JitterRetry(total=3)

    # Mock the parent's get_backoff_time to return a fixed value
    with patch.object(Retry, "get_backoff_time", return_value=1.0):
        # Test multiple times to ensure jitter is added
        backoff_times = [retry.get_backoff_time() for _ in range(100)]

        # All backoff times should be between 1.0 and 1.25
        assert all(1.0 <= t <= 1.25 for t in backoff_times)
        # Values should be different (jitter is working)
        assert len(set(backoff_times)) > 1


def test_jitter_retry_behavior():
    """Test that retries actually occur and respect the total count."""
    retry = JitterRetry(total=3)
    mock_response = MagicMock()
    mock_response.status = 500  # Simulate server error

    # Simulate a failing request
    with pytest.raises(MaxRetryError) as exc_info:
        retry.increment(method="GET", url="http://test.com", response=mock_response, error=None)

    # Verify the error contains the expected information
    assert "Max retries exceeded" in str(exc_info.value)
    assert exc_info.value.reason.status == 500


def test_jitter_retry_success():
    """Test that retry stops on successful response."""
    retry = JitterRetry(total=3)
    mock_response = MagicMock()
    mock_response.status = 200  # Success response

    # Should not raise an exception for successful response
    new_retry = retry.increment(
        method="GET", url="http://test.com", response=mock_response, error=None
    )

    # Verify retry count is decremented
    assert new_retry.remaining == retry.total - 1
