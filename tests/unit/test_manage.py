import pytest
import pinecone
import time

class TestManage:
    
    def test_get_api_instance_without_host(self):
        pinecone.init(api_key="123-456-789", environment="my-environment")
        api_instance = pinecone.manage._get_api_instance()
        assert api_instance.api_client.configuration.host == "https://controller.my-environment.pinecone.io"

    def test_get_api_instance_with_host(self):
        pinecone.init(api_key="123-456-789", environment="my-environment", host="my-host")
        api_instance = pinecone.manage._get_api_instance()
        assert api_instance.api_client.configuration.host == "my-host"

    @pytest.mark.parametrize("timeout_value, get_status_calls, time_sleep_calls, get_status_responses", [
        # No timeout, _get_status called twice, sleep called once
        (None, 2, 1, [{"ready": False}, {"ready": True}]),
        # Timeout of 10 seconds, _get_status called 3 times, sleep twice
        (10, 3, 2, [{"ready": False}, {"ready": False}, {"ready": True}]),
        # Timeout of -1 seconds, _get_status not called, no sleep
        (-1, 0, 0, [{"ready": False}]),
    ])
    def test_create_index_with_timeout(self, mocker, timeout_value, get_status_calls, time_sleep_calls, get_status_responses):
        mocker.patch('pinecone.manage._get_api_instance', return_value=mocker.Mock())
        mocker.patch('pinecone.manage._get_status', side_effect=get_status_responses)
        mocker.patch('time.sleep')

        pinecone.manage.create_index("my-index", 10, timeout=timeout_value)

        pinecone.manage._get_api_instance.assert_called_once()
        assert pinecone.manage._get_status.call_count == get_status_calls
        assert time.sleep.call_count == time_sleep_calls

    @pytest.mark.parametrize("timeout_value, list_indexes_calls, time_sleep_calls, list_indexes_responses", [
        # No timeout, list_indexes called twice, sleep called once
        (None, 2, 1, [["my-index", "index-1"], ["index-1"]]),
        # Timeout of 10 seconds, list_indexes called 3 times, sleep twice
        (10, 3, 2, [["my-index", "index-1"], ["my-index", "index-1"], ["index-1"]]),
        # Timeout of -1 seconds, list_indexes not called, no sleep
        (-1, 0, 0, [["my-index", "index-1"]]),
    ])
    def test_delete_index_with_timeout(self, mocker, timeout_value, list_indexes_calls, time_sleep_calls, list_indexes_responses):
        api_instance_mock = mocker.Mock()
        api_instance_mock.list_indexes = mocker.Mock(side_effect=list_indexes_responses)
        mocker.patch('pinecone.manage._get_api_instance', return_value=api_instance_mock)
        mocker.patch('time.sleep')

        pinecone.manage.delete_index("my-index", timeout=timeout_value)

        pinecone.manage._get_api_instance.assert_called_once()
        assert api_instance_mock.list_indexes.call_count == list_indexes_calls
        assert time.sleep.call_count == time_sleep_calls