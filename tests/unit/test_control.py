import pytest
from pinecone import Pinecone
from pinecone.core.client.api import IndexOperationsApi
import time

class TestControl:
    def test_default_host(self):
        p = Pinecone(api_key="123-456-789")
        assert p.index_api.api_client.configuration.host == "https://api.pinecone.io"

    def test_passing_host(self):
        p = Pinecone(api_key="123-456-789", host="my-host")
        assert p.index_api.api_client.configuration.host == "my-host"


    @pytest.mark.parametrize("timeout_value, describe_index_responses, expected_describe_index_calls, expected_sleep_calls", [
        # When timeout=None, describe_index is called until ready
        (None, [{ "status": {"ready": False}}, {"status": {"ready": True}}], 2, 1),

        # Timeout of 10 seconds, describe_index called 3 times, sleep twice
        (10, [{"status": {"ready": False}}, {"status": {"ready": False}}, {"status": {"ready": True}}], 3, 2),

        # When timeout=-1, create_index returns immediately without calling describe_index or sleep
        (-1, [{"status": {"ready": False}}], 0, 0),
    ])
    def test_create_index_with_timeout(self, mocker, timeout_value, describe_index_responses, expected_describe_index_calls, expected_sleep_calls):
        mocker.patch.object(IndexOperationsApi, 'describe_index', side_effect=describe_index_responses)
        mocker.patch.object(IndexOperationsApi, 'create_index')
        mocker.patch('time.sleep')

        p = Pinecone(api_key="123-456-789")
        p.create_index("my-index", 10, timeout=timeout_value, cloud="aws", region="us-west1", capacity_mode="pod")

        assert IndexOperationsApi.create_index.call_count == 1
        assert IndexOperationsApi.describe_index.call_count == expected_describe_index_calls
        assert time.sleep.call_count == expected_sleep_calls

    def test_create_index_when_timeout_exceeded(self, mocker):
        with pytest.raises(TimeoutError):
            get_status_responses = [{"status": {"ready": False}}] * 5
            mocker.patch.object(IndexOperationsApi, 'describe_index', side_effect=get_status_responses)
            mocker.patch.object(IndexOperationsApi, 'create_index')
            mocker.patch('time.sleep')

            p = Pinecone(api_key="123-456-789")
            p.create_index("my-index", 10, timeout=10, cloud="aws", region="us-west1", capacity_mode="pod")

    # @pytest.mark.parametrize("timeout_value, list_indexes_calls, time_sleep_calls, list_indexes_responses", [
    #     # No timeout, list_indexes called twice, sleep called once
    #     (None, 2, 1, [["my-index", "index-1"], ["index-1"]]),
    #     # Timeout of 10 seconds, list_indexes called 3 times, sleep twice
    #     (10, 3, 2, [["my-index", "index-1"], ["my-index", "index-1"], ["index-1"]]),
    #     # Timeout of -1 seconds, list_indexes not called, no sleep
    #     (-1, 0, 0, [["my-index", "index-1"]]),
    # ])
    # def test_delete_index_with_timeout(self, mocker, timeout_value, list_indexes_calls, time_sleep_calls, list_indexes_responses):
    #     api_instance_mock = mocker.Mock()
    #     api_instance_mock.list_indexes = mocker.Mock(side_effect=list_indexes_responses)
    #     mocker.patch('pinecone.manage._get_api_instance', return_value=api_instance_mock)
    #     mocker.patch('time.sleep')

    #     pinecone.manage.delete_index("my-index", timeout=timeout_value)

    #     pinecone.manage._get_api_instance.assert_called_once()
    #     assert api_instance_mock.list_indexes.call_count == list_indexes_calls
    #     assert time.sleep.call_count == time_sleep_calls