import pytest
from unittest.mock import patch, MagicMock
import urllib3
from urllib3.exceptions import MaxRetryError
from pinecone.openapi_support.rest_urllib3 import Urllib3RestClient
from pinecone.config.openapi_configuration import Configuration


class TestUrllib3RestClient:
    @pytest.fixture
    def config(self):
        return Configuration(api_key="test-key")

    @pytest.fixture
    def client(self, config):
        return Urllib3RestClient(config)

    def test_retry_on_500_error(self, client):
        # Mock response that fails with 500
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.data = b'{"error": "Internal Server Error"}'
        mock_response.headers = {}
        mock_response.reason = "Internal Server Error"

        # Mock pool manager to fail twice then succeed
        with patch.object(client.pool_manager, "request") as mock_request:
            mock_request.side_effect = [
                urllib3.exceptions.HTTPError(response=mock_response),
                urllib3.exceptions.HTTPError(response=mock_response),
                mock_response,  # Success on third try
            ]

            # Make request
            response = client.request(
                method="GET",
                url="https://api.pinecone.io/test",
                headers={"Authorization": "test-key"},
            )

            # Verify request was made 3 times (initial + 2 retries)
            assert mock_request.call_count == 3

            # Verify the response is successful
            assert response.status == 200

    def test_max_retries_exceeded(self, client):
        # Mock response that always fails with 500
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.data = b'{"error": "Internal Server Error"}'
        mock_response.headers = {}
        mock_response.reason = "Internal Server Error"

        # Mock pool manager to always fail
        with patch.object(client.pool_manager, "request") as mock_request:
            mock_request.side_effect = urllib3.exceptions.HTTPError(response=mock_response)

            # Make request and expect MaxRetryError
            with pytest.raises(MaxRetryError):
                client.request(
                    method="GET",
                    url="https://api.pinecone.io/test",
                    headers={"Authorization": "test-key"},
                )

            # Verify request was made 4 times (initial + 3 retries)
            assert mock_request.call_count == 4

    def test_custom_retry_config(self):
        # Create custom retry configuration
        custom_retry = urllib3.Retry(
            total=2, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)
        )

        config = Configuration(api_key="test-key", retries=custom_retry)
        client = Urllib3RestClient(config)

        # Mock response that fails with 500
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.data = b'{"error": "Internal Server Error"}'
        mock_response.headers = {}
        mock_response.reason = "Internal Server Error"

        # Mock pool manager to fail once then succeed
        with patch.object(client.pool_manager, "request") as mock_request:
            mock_request.side_effect = [
                urllib3.exceptions.HTTPError(response=mock_response),
                mock_response,  # Success on second try
            ]

            # Make request
            response = client.request(
                method="GET",
                url="https://api.pinecone.io/test",
                headers={"Authorization": "test-key"},
            )

            # Verify request was made 2 times (initial + 1 retry)
            assert mock_request.call_count == 2

            # Verify the response is successful
            assert response.status == 200
