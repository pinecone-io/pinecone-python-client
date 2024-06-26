import pytest
import re
from unittest.mock import patch, MagicMock
from pinecone import ConfigBuilder, Pinecone, PodSpec, ServerlessSpec
from pinecone.core.control.client.models import IndexList, IndexModel
from pinecone.core.control.client.api.manage_indexes_api import ManageIndexesApi
from pinecone.core.control.client.configuration import Configuration as OpenApiConfiguration

import time


@pytest.fixture
def index_list_response():
    return IndexList(
        indexes=[
            IndexModel(
                name="index1",
                dimension=10,
                metric="euclidean",
                host="asdf",
                status={"ready": True},
                spec={},
                _check_type=False,
            ),
            IndexModel(
                name="index2",
                dimension=10,
                metric="euclidean",
                host="asdf",
                status={"ready": True},
                spec={},
                _check_type=False,
            ),
            IndexModel(
                name="index3",
                dimension=10,
                metric="euclidean",
                host="asdf",
                status={"ready": True},
                spec={},
                _check_type=False,
            ),
        ]
    )


class TestControl:
    def test_plugins_are_installed(self):
        with patch("pinecone.control.pinecone.install_plugins") as mock_install_plugins:
            p = Pinecone(api_key="asdf")
            mock_install_plugins.assert_called_once()

    def test_bad_plugin_doesnt_break_sdk(self):
        with patch("pinecone.control.pinecone.install_plugins", side_effect=Exception("bad plugin")):
            try:
                p = Pinecone(api_key="asdf")
            except Exception as e:
                assert False, f"Unexpected exception: {e}"

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
        assert "User-Agent" in p.index_api.api_client.default_headers
        assert len(p.index_api.api_client.default_headers) == 3

    def test_overwrite_useragent(self):
        # This doesn't seem like a common use case, but we may want to allow this
        # when embedding the client in other pinecone tools such as canopy.
        extras = {"User-Agent": "test-user-agent"}
        p = Pinecone(api_key="123-456-789", additional_headers=extras)
        assert p.index_api.api_client.default_headers["User-Agent"] == "test-user-agent"
        assert len(p.index_api.api_client.default_headers) == 1

    def test_set_source_tag_in_useragent(self):
        p = Pinecone(api_key="123-456-789", source_tag="test_source_tag")
        assert re.search(r"source_tag=test_source_tag", p.index_api.api_client.user_agent) is not None

    def test_set_source_tag_in_useragent_via_config(self):
        config = ConfigBuilder.build(api_key="YOUR_API_KEY", host="https://my-host", source_tag="my_source_tag")
        p = Pinecone(config=config)
        assert re.search(r"source_tag=my_source_tag", p.index_api.api_client.user_agent) is not None

    @pytest.mark.parametrize(
        "timeout_value, describe_index_responses, expected_describe_index_calls, expected_sleep_calls",
        [
            # When timeout=None, describe_index is called until ready
            (None, [{"status": {"ready": False}}, {"status": {"ready": True}}], 2, 1),
            # Timeout of 10 seconds, describe_index called 3 times, sleep twice
            (10, [{"status": {"ready": False}}, {"status": {"ready": False}}, {"status": {"ready": True}}], 3, 2),
            # When timeout=-1, create_index returns immediately without calling describe_index or sleep
            (-1, [{"status": {"ready": False}}], 0, 0),
        ],
    )
    def test_create_index_with_timeout(
        self, mocker, timeout_value, describe_index_responses, expected_describe_index_calls, expected_sleep_calls
    ):
        p = Pinecone(api_key="123-456-789")
        mocker.patch.object(p.index_api, "describe_index", side_effect=describe_index_responses)
        mocker.patch.object(p.index_api, "create_index")
        mocker.patch("time.sleep")

        p.create_index(
            name="my-index", dimension=10, spec=ServerlessSpec(cloud="aws", region="us-west1"), timeout=timeout_value
        )

        assert p.index_api.create_index.call_count == 1
        assert p.index_api.describe_index.call_count == expected_describe_index_calls
        assert time.sleep.call_count == expected_sleep_calls

    @pytest.mark.parametrize('index_spec', [
        {"serverless": {"cloud": "aws", "region": "us-west1"}},
        {"pod": {"environment": "us-west1-gcp", "pod_type": "p1.x1", "pods": 1, "replicas": 1, "shards": 1}},
    ])
    def test_create_index(self, mocker, index_spec):
        p = Pinecone(api_key="123-456-789")

        mock_api = MagicMock()
        mocker.patch.object(p, "index_api", mock_api)
        
        p.create_index(name="my-index", dimension=10, spec=index_spec)

        mock_api.create_index.assert_called_once()

    @pytest.mark.parametrize(
        "timeout_value, describe_index_responses, expected_describe_index_calls, expected_sleep_calls",
        [
            # When timeout=None, describe_index is called until ready
            (None, [{"status": {"ready": False}}, {"status": {"ready": True}}], 2, 1),
            # Timeout of 10 seconds, describe_index called 3 times, sleep twice
            (10, [{"status": {"ready": False}}, {"status": {"ready": False}}, {"status": {"ready": True}}], 3, 2),
            # When timeout=-1, create_index returns immediately without calling describe_index or sleep
            (-1, [{"status": {"ready": False}}], 0, 0),
        ],
    )
    def test_create_index_from_source_collection(
        self, mocker, timeout_value, describe_index_responses, expected_describe_index_calls, expected_sleep_calls
    ):
        p = Pinecone(api_key="123-456-789")
        mocker.patch.object(p.index_api, "describe_index", side_effect=describe_index_responses)
        mocker.patch.object(p.index_api, "create_index")
        mocker.patch("time.sleep")

        p.create_index(
            name="my-index",
            dimension=10,
            spec=PodSpec(environment="us-east1-gcp", source_collection="my-collection"),
            timeout=timeout_value,
        )

        assert p.index_api.create_index.call_count == 1
        assert p.index_api.describe_index.call_count == expected_describe_index_calls
        assert time.sleep.call_count == expected_sleep_calls

    def test_create_index_when_timeout_exceeded(self, mocker):
        with pytest.raises(TimeoutError):
            p = Pinecone(api_key="123-456-789")
            mocker.patch.object(p.index_api, "create_index")

            describe_index_response = [{"status": {"ready": False}}] * 5
            mocker.patch.object(p.index_api, "describe_index", side_effect=describe_index_response)
            mocker.patch("time.sleep")

            p.create_index(name="my-index", dimension=10, timeout=10, spec=PodSpec(environment="us-west1-gcp"))

    def test_list_indexes_returns_iterable(self, mocker, index_list_response):
        p = Pinecone(api_key="123-456-789")

        mocker.patch.object(p.index_api, "list_indexes", side_effect=[index_list_response])

        response = p.list_indexes()
        assert [i.name for i in response] == ["index1", "index2", "index3"]

    def test_api_key_and_openapi_config(self, mocker):
        p = Pinecone(api_key="123", openapi_config=OpenApiConfiguration.get_default_copy())
        assert p.config.api_key == "123"

class TestIndexConfig:
    def test_default_pool_threads(self):
        pc = Pinecone(api_key="123-456-789")
        index = pc.Index(host="my-host.svg.pinecone.io")
        assert index._vector_api.api_client.pool_threads == 1

    def test_pool_threads_when_indexapi_passed(self):
        pc = Pinecone(api_key="123-456-789", pool_threads=2, index_api=ManageIndexesApi())
        index = pc.Index(host="my-host.svg.pinecone.io")
        assert index._vector_api.api_client.pool_threads == 2

    def test_target_index_with_pool_threads_inherited(self):
        pc = Pinecone(api_key="123-456-789", pool_threads=10, foo="bar")
        index = pc.Index(host="my-host.svg.pinecone.io")
        assert index._vector_api.api_client.pool_threads == 10

    def test_target_index_with_pool_threads_kwarg(self):
        pc = Pinecone(api_key="123-456-789", pool_threads=10)
        index = pc.Index(host="my-host.svg.pinecone.io", pool_threads=5)
        assert index._vector_api.api_client.pool_threads == 5
