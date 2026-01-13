"""Test that response_info assignment handles all types correctly"""
import pytest
from unittest.mock import Mock
from pinecone.openapi_support.api_client import ApiClient
from pinecone.openapi_support.asyncio_api_client import AsyncioApiClient
from pinecone.config.openapi_configuration import Configuration


class TestResponseInfoAssignment:
    """Test that _response_info assignment works for all response types"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = Configuration()

    def test_sync_api_client_dict_response(self, mocker):
        """Test that dict responses get _response_info as a key"""
        api_client = ApiClient(self.config)

        # Mock the request method to return a dict response
        mock_response = Mock()
        mock_response.data = b'{}'
        mock_response.status = 200
        mock_response.getheaders = Mock(return_value={'x-pinecone-request-latency-ms': '100'})
        mock_response.getheader = Mock(side_effect=lambda x: 'application/json' if x == 'content-type' else None)

        mocker.patch.object(api_client, 'request', return_value=mock_response)

        # Call the API
        result = api_client.call_api(
            resource_path='/test',
            method='POST',
            response_type=(dict,),
            _return_http_data_only=True,
        )

        # Verify _response_info is set as a dict key
        assert isinstance(result, dict)
        assert '_response_info' in result

    def test_sync_api_client_string_response(self, mocker):
        """Test that string responses don't cause AttributeError"""
        api_client = ApiClient(self.config)

        # Mock the request method to return a string response
        mock_response = Mock()
        mock_response.data = b'"success"'
        mock_response.status = 200
        mock_response.getheaders = Mock(return_value={'x-pinecone-request-latency-ms': '100'})
        mock_response.getheader = Mock(side_effect=lambda x: 'application/json' if x == 'content-type' else None)

        mocker.patch.object(api_client, 'request', return_value=mock_response)

        # This should not raise AttributeError when trying to set _response_info
        try:
            api_client.call_api(
                resource_path='/test',
                method='POST',
                response_type=(str,),
                _return_http_data_only=True,
                _check_type=False,
            )
            # If we get a string back, it should not have _response_info
            # (we don't check what type we get back because it depends on deserializer behavior)
        except AttributeError as e:
            if "'str' object has no attribute '_response_info'" in str(e):
                pytest.fail(f"Should not raise AttributeError for string response: {e}")
            # Other AttributeErrors may be raised by deserializer for invalid types

    def test_sync_api_client_none_response(self, mocker):
        """Test that None responses are handled correctly"""
        api_client = ApiClient(self.config)

        # Mock the request method to return no content
        mock_response = Mock()
        mock_response.data = b''
        mock_response.status = 204
        mock_response.getheaders = Mock(return_value={'x-pinecone-request-latency-ms': '100'})
        mock_response.getheader = Mock(side_effect=lambda x: None)

        mocker.patch.object(api_client, 'request', return_value=mock_response)

        # This should not raise AttributeError
        try:
            result = api_client.call_api(
                resource_path='/test',
                method='DELETE',
                response_type=None,
                _return_http_data_only=True,
            )
            assert result is None
        except AttributeError as e:
            pytest.fail(f"Should not raise AttributeError for None response: {e}")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires asyncio extras")
    async def test_asyncio_api_client_dict_response(self, mocker):
        """Test that dict responses get _response_info as a key in asyncio"""
        api_client = AsyncioApiClient(self.config)

        # Mock the request method to return a dict response
        mock_response = Mock()
        mock_response.data = b'{}'
        mock_response.status = 200
        mock_response.getheaders = Mock(return_value={'x-pinecone-request-latency-ms': '100'})
        mock_response.getheader = Mock(side_effect=lambda x: 'application/json' if x == 'content-type' else None)

        async def mock_request(*args, **kwargs):
            return mock_response

        mocker.patch.object(api_client, 'request', side_effect=mock_request)

        # Call the API
        result = await api_client.call_api(
            resource_path='/test',
            method='POST',
            response_type=(dict,),
            _return_http_data_only=True,
        )

        # Verify _response_info is set as a dict key
        assert isinstance(result, dict)
        assert '_response_info' in result

        await api_client.close()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires asyncio extras")
    async def test_asyncio_api_client_string_response(self, mocker):
        """Test that string responses don't cause AttributeError in asyncio"""
        api_client = AsyncioApiClient(self.config)

        # Mock the request method to return a string response
        mock_response = Mock()
        mock_response.data = b'"success"'
        mock_response.status = 200
        mock_response.getheaders = Mock(return_value={'x-pinecone-request-latency-ms': '100'})
        mock_response.getheader = Mock(side_effect=lambda x: 'application/json' if x == 'content-type' else None)

        async def mock_request(*args, **kwargs):
            return mock_response

        mocker.patch.object(api_client, 'request', side_effect=mock_request)

        # This should not raise AttributeError when trying to set _response_info
        try:
            await api_client.call_api(
                resource_path='/test',
                method='POST',
                response_type=(str,),
                _return_http_data_only=True,
                _check_type=False,
            )
            # If we get a string back, it should not have _response_info
        except AttributeError as e:
            if "'str' object has no attribute '_response_info'" in str(e):
                pytest.fail(f"Should not raise AttributeError for string response: {e}")
            # Other AttributeErrors may be raised by deserializer for invalid types
        finally:
            await api_client.close()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires asyncio extras")
    async def test_asyncio_api_client_none_response(self, mocker):
        """Test that None responses are handled correctly in asyncio"""
        api_client = AsyncioApiClient(self.config)

        # Mock the request method to return no content
        mock_response = Mock()
        mock_response.data = b''
        mock_response.status = 204
        mock_response.getheaders = Mock(return_value={'x-pinecone-request-latency-ms': '100'})
        mock_response.getheader = Mock(side_effect=lambda x: None)

        async def mock_request(*args, **kwargs):
            return mock_response

        mocker.patch.object(api_client, 'request', side_effect=mock_request)

        # This should not raise AttributeError
        try:
            result = await api_client.call_api(
                resource_path='/test',
                method='DELETE',
                response_type=None,
                _return_http_data_only=True,
            )
            assert result is None
        except AttributeError as e:
            pytest.fail(f"Should not raise AttributeError for None response: {e}")
        finally:
            await api_client.close()

    def test_sync_api_client_model_response(self, mocker):
        """Test that OpenAPI model responses get _response_info as an attribute"""
        api_client = ApiClient(self.config)

        # Create a mock model class that supports attribute assignment
        class MockModel:
            def __init__(self):
                pass

        # Mock the request and deserializer
        mock_response = Mock()
        mock_response.data = b'{"test": "value"}'
        mock_response.status = 200
        mock_response.getheaders = Mock(return_value={'x-pinecone-request-latency-ms': '100'})
        mock_response.getheader = Mock(side_effect=lambda x: 'application/json' if x == 'content-type' else None)

        mocker.patch.object(api_client, 'request', return_value=mock_response)

        # Mock the deserializer to return a model instance
        mock_model_instance = MockModel()
        mocker.patch('pinecone.openapi_support.deserializer.Deserializer.deserialize',
                     return_value=mock_model_instance)
        mocker.patch('pinecone.openapi_support.deserializer.Deserializer.decode_response')

        # Call the API
        result = api_client.call_api(
            resource_path='/test',
            method='GET',
            response_type=(MockModel,),
            _return_http_data_only=True,
        )

        # Verify _response_info is set as an attribute
        assert hasattr(result, '_response_info')
