import pytest
import re
from unittest.mock import patch, MagicMock
from pinecone import (
    Pinecone,
    PodSpec,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    GcpRegion,
    PodIndexEnvironment,
    PodType,
)
from pinecone.core.openapi.db_control.models import (
    IndexList,
    IndexModel,
    DeletionProtection,
    IndexModelSpec,
    ServerlessSpec as ServerlessSpecOpenApi,
    IndexModelStatus,
)
from pinecone.utils import PluginAware


import time


def description_with_status(status: bool):
    state = "Ready" if status else "Initializing"
    return IndexModel(
        name="foo",
        status=IndexModelStatus(ready=status, state=state),
        dimension=10,
        deletion_protection=DeletionProtection(value="enabled"),
        host="https://foo.pinecone.io",
        metric="euclidean",
        spec=IndexModelSpec(serverless=ServerlessSpecOpenApi(cloud="aws", region="us-west1")),
    )


@pytest.fixture
def index_list_response():
    return IndexList(
        indexes=[
            IndexModel(
                name="index1",
                dimension=10,
                metric="euclidean",
                host="asdf.pinecone.io",
                status={"ready": True},
                spec={},
                deletion_protection=DeletionProtection("enabled"),
                _check_type=False,
            ),
            IndexModel(
                name="index2",
                dimension=10,
                metric="euclidean",
                host="asdf.pinecone.io",
                status={"ready": True},
                spec={},
                deletion_protection=DeletionProtection("enabled"),
                _check_type=False,
            ),
            IndexModel(
                name="index3",
                dimension=10,
                metric="euclidean",
                host="asdf.pinecone.io",
                status={"ready": True},
                spec={},
                deletion_protection=DeletionProtection("disabled"),
                _check_type=False,
            ),
        ]
    )


class TestControl:
    def test_plugins_are_installed(self):
        with patch.object(PluginAware, "load_plugins") as mock_install_plugins:
            Pinecone(api_key="asdf")
            mock_install_plugins.assert_called_once()

    def test_default_host(self):
        p = Pinecone(api_key="123-456-789")
        assert p.index_api.api_client.configuration.host == "https://api.pinecone.io"

    def test_passing_host(self):
        p = Pinecone(api_key="123-456-789", host="my-host.pinecone.io")
        assert p.index_api.api_client.configuration.host == "https://my-host.pinecone.io"

    def test_passing_additional_headers(self):
        extras = {"header1": "my-value", "header2": "my-value2"}
        p = Pinecone(api_key="123-456-789", additional_headers=extras)

        for key, value in extras.items():
            assert p.index_api.api_client.default_headers[key] == value
        assert "User-Agent" in p.index_api.api_client.default_headers
        assert "X-Pinecone-API-Version" in p.index_api.api_client.default_headers
        assert "header1" in p.index_api.api_client.default_headers
        assert "header2" in p.index_api.api_client.default_headers
        assert len(p.index_api.api_client.default_headers) == 4

    def test_overwrite_useragent(self):
        # This doesn't seem like a common use case, but we may want to allow this
        # when embedding the client in other pinecone tools such as canopy.
        extras = {"User-Agent": "test-user-agent"}
        p = Pinecone(api_key="123-456-789", additional_headers=extras)
        assert "X-Pinecone-API-Version" in p.index_api.api_client.default_headers
        assert p.index_api.api_client.default_headers["User-Agent"] == "test-user-agent"
        assert len(p.index_api.api_client.default_headers) == 2

    def test_set_source_tag_in_useragent(self):
        p = Pinecone(api_key="123-456-789", source_tag="test_source_tag")
        assert (
            re.search(r"source_tag=test_source_tag", p.index_api.api_client.user_agent) is not None
        )

    @pytest.mark.parametrize(
        "timeout_value, describe_index_responses, expected_describe_index_calls, expected_sleep_calls",
        [
            # When timeout=None, describe_index is called until ready
            (None, [description_with_status(False), description_with_status(True)], 2, 1),
            # # Timeout of 10 seconds, describe_index called 3 times, sleep twice
            (
                10,
                [
                    description_with_status(False),
                    description_with_status(False),
                    description_with_status(True),
                ],
                3,
                2,
            ),
            # # When timeout=-1, create_index returns immediately without calling sleep
            (-1, [description_with_status(False)], 0, 0),
        ],
    )
    def test_create_index_with_timeout(
        self,
        mocker,
        timeout_value,
        describe_index_responses,
        expected_describe_index_calls,
        expected_sleep_calls,
    ):
        p = Pinecone(api_key="123-456-789")
        mocker.patch.object(p.index_api, "describe_index", side_effect=describe_index_responses)
        mocker.patch.object(p.index_api, "create_index")
        mocker.patch("time.sleep")

        p.create_index(
            name="my-index",
            dimension=10,
            spec=ServerlessSpec(cloud="aws", region="us-west1"),
            timeout=timeout_value,
        )

        assert p.index_api.create_index.call_count == 1
        assert p.index_api.describe_index.call_count == expected_describe_index_calls
        assert time.sleep.call_count == expected_sleep_calls

    @pytest.mark.parametrize(
        "index_spec",
        [
            ServerlessSpec(cloud="aws", region="us-west-2"),
            ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_WEST_2),
            ServerlessSpec(cloud=CloudProvider.AWS, region="us-west-2"),
            ServerlessSpec(cloud="aws", region="us-west-2"),
            ServerlessSpec(cloud="aws", region="unknown-region"),
            ServerlessSpec(cloud=CloudProvider.GCP, region=GcpRegion.US_CENTRAL1),
            {"serverless": {"cloud": "aws", "region": "us-west1"}},
            {"serverless": {"cloud": "aws", "region": "us-west1", "uknown_key": "value"}},
            PodSpec(environment="us-west1-gcp", pod_type="p1.x1"),
            PodSpec(environment=PodIndexEnvironment.US_WEST1_GCP, pod_type=PodType.P2_X2),
            PodSpec(environment=PodIndexEnvironment.US_WEST1_GCP, pod_type="s1.x4"),
            PodSpec(environment=PodIndexEnvironment.US_EAST1_AWS, pod_type="unknown-pod-type"),
            PodSpec(environment="us-west1-gcp", pod_type="p1.x1", pods=2, replicas=1, shards=1),
            {"pod": {"environment": "us-west1-gcp", "pod_type": "p1.x1"}},
            {"pod": {"environment": "us-west1-gcp", "pod_type": "p1.x1", "unknown_key": "value"}},
            {
                "pod": {
                    "environment": "us-west1-gcp",
                    "pod_type": "p1.x1",
                    "pods": 2,
                    "replicas": 1,
                    "shards": 1,
                    "metadata_config": {"indexed": ["foo"]},
                    "source_collection": "bar",
                }
            },
            {
                "pod": {
                    "environment": "us-west1-gcp",
                    "pod_type": "p1.x1",
                    "pods": None,
                    "replicas": None,
                    "shards": None,
                    "metadata_config": None,
                    "source_collection": None,
                }
            },
        ],
    )
    def test_create_index_with_spec_dictionary(self, mocker, index_spec):
        p = Pinecone(api_key="123-456-789")

        mock_api = MagicMock()
        mocker.patch.object(p, "index_api", mock_api)

        p.create_index(name="my-index", dimension=10, spec=index_spec)

        mock_api.create_index.assert_called_once()

    @pytest.mark.parametrize(
        "timeout_value, describe_index_responses, expected_describe_index_calls, expected_sleep_calls",
        [
            # When timeout=None, describe_index is called until ready
            (None, [description_with_status(False), description_with_status(True)], 2, 1),
            # Timeout of 10 seconds, describe_index called 3 times, sleep twice
            (
                10,
                [
                    description_with_status(False),
                    description_with_status(False),
                    description_with_status(True),
                ],
                3,
                2,
            ),
            # When timeout=-1, create_index returns immediately without calling sleep
            (-1, [description_with_status(False)], 0, 0),
        ],
    )
    def test_create_index_from_source_collection(
        self,
        mocker,
        timeout_value,
        describe_index_responses,
        expected_describe_index_calls,
        expected_sleep_calls,
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

            describe_index_response = [description_with_status(False)] * 5
            mocker.patch.object(p.index_api, "describe_index", side_effect=describe_index_response)
            mocker.patch("time.sleep")

            p.create_index(
                name="my-index", dimension=10, timeout=10, spec=PodSpec(environment="us-west1-gcp")
            )

    def test_list_indexes_returns_iterable(self, mocker, index_list_response):
        p = Pinecone(api_key="123-456-789")

        mocker.patch.object(p.index_api, "list_indexes", side_effect=[index_list_response])

        response = p.list_indexes()
        assert [i.name for i in response] == ["index1", "index2", "index3"]


class TestIndexConfig:
    def test_default_pool_threads(self):
        pc = Pinecone(api_key="123-456-789")
        index = pc.Index(host="my-host.svg.pinecone.io")
        assert index._vector_api.api_client.pool_threads >= 1

    def test_target_index_with_pool_threads_inherited(self):
        pc = Pinecone(api_key="123-456-789", pool_threads=10, foo="bar")
        index = pc.Index(host="my-host.svg.pinecone.io")
        assert index._vector_api.api_client.pool_threads == 10

    def test_target_index_with_pool_threads_kwarg(self):
        pc = Pinecone(api_key="123-456-789", pool_threads=10)
        index = pc.Index(host="my-host.svg.pinecone.io", pool_threads=5)
        assert index._vector_api.api_client.pool_threads == 5
