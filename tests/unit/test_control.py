import pytest
from pinecone import Pinecone, PodSpec, ServerlessSpec
from pinecone.core.client.models import IndexList, IndexModel
from pinecone.core.client.api.manage_indexes_api import ManageIndexesApi
import time

@pytest.fixture
def index_list_response():
    return IndexList(indexes=[
        IndexModel(name="index1", dimension=10, metric="euclidean", host="asdf", status={"ready": True}, spec={}, _check_type=False),
        IndexModel(name="index2", dimension=10, metric="euclidean", host="asdf", status={"ready": True}, spec={}, _check_type=False),
        IndexModel(name="index3", dimension=10, metric="euclidean", host="asdf", status={"ready": True}, spec={}, _check_type=False),

    ])

class TestControl:
    def test_default_host(self):
        p = Pinecone(api_key="123-456-789")
        assert p.index_api.api_client.configuration.host == "https://api.pinecone.io"

    def test_passing_host(self):
        p = Pinecone(api_key="123-456-789", host="my-host")
        assert p.index_api.api_client.configuration.host == "https://my-host"

    def test_passing_additional_headers(self):
        extras = {"header1": "my-value", "header2": "my-value2"}
        p = Pinecone(api_key="123-456-789", additional_headers=extras)

        for key, value in extras.items():
            assert p.index_api.api_client.default_headers[key] == value

    @pytest.mark.parametrize("timeout_value, describe_index_responses, expected_describe_index_calls, expected_sleep_calls", [
        # When timeout=None, describe_index is called until ready
        (None, [{ "status": {"ready": False}}, {"status": {"ready": True}}], 2, 1),

        # Timeout of 10 seconds, describe_index called 3 times, sleep twice
        (10, [{"status": {"ready": False}}, {"status": {"ready": False}}, {"status": {"ready": True}}], 3, 2),

        # When timeout=-1, create_index returns immediately without calling describe_index or sleep
        (-1, [{"status": {"ready": False}}], 0, 0),
    ])
    def test_create_index_with_timeout(self, mocker, timeout_value, describe_index_responses, expected_describe_index_calls, expected_sleep_calls):
        p = Pinecone(api_key="123-456-789")
        mocker.patch.object(p.index_api, 'describe_index', side_effect=describe_index_responses)
        mocker.patch.object(p.index_api, 'create_index')
        mocker.patch('time.sleep')

        p.create_index(name="my-index", dimension=10, spec=ServerlessSpec(cloud="aws", region="us-west1"), timeout=timeout_value)

        assert p.index_api.create_index.call_count == 1
        assert p.index_api.describe_index.call_count == expected_describe_index_calls
        assert time.sleep.call_count == expected_sleep_calls

    def test_create_index_when_timeout_exceeded(self, mocker):
        with pytest.raises(TimeoutError):
            p = Pinecone(api_key="123-456-789")
            mocker.patch.object(p.index_api, 'create_index')

            describe_index_response = [{"status": {"ready": False}}] * 5
            mocker.patch.object(p.index_api, 'describe_index', side_effect=describe_index_response)
            mocker.patch('time.sleep')

            p.create_index(name="my-index", dimension=10, timeout=10, spec=PodSpec(environment="us-west1-gcp"))

    def test_list_indexes_returns_iterable(self, mocker, index_list_response):
        p = Pinecone(api_key="123-456-789")
       
        mocker.patch.object(p.index_api, 'list_indexes', side_effect=[index_list_response])

        response = p.list_indexes()
        assert [i.name for i in response] == ["index1", "index2", "index3"]


class TestIndexConfig:
    def test_default_pool_threads(self):
        pc = Pinecone(api_key="123-456-789")
        index = pc.Index(host='my-host.svg.pinecone.io')
        assert index._api_client.pool_threads == 1

    def test_pool_threads_when_indexapi_passed(self):
        pc = Pinecone(api_key="123-456-789", pool_threads=2, index_api=ManageIndexesApi())
        index = pc.Index(host='my-host.svg.pinecone.io')
        assert index._api_client.pool_threads == 2

    def test_target_index_with_pool_threads_inherited(self):
        pc = Pinecone(api_key="123-456-789", pool_threads=10, foo='bar')
        index = pc.Index(host='my-host.svg.pinecone.io')
        assert index._api_client.pool_threads == 10

    def test_target_index_with_pool_threads_kwarg(self):
        pc = Pinecone(api_key="123-456-789", pool_threads=10)
        index = pc.Index(host='my-host.svg.pinecone.io', pool_threads=5)
        assert index._api_client.pool_threads == 5

