import pytest
import pinecone
import os
import tests.test_helpers as test_helpers


class TestConfigureIndex:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # setup
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")

        self.index_name = test_helpers.generate_index_name()
        self.pinecone = pinecone
        self.pinecone.init(api_key=api_key, environment=environment)
        self.pinecone.create_index(name=self.index_name, dimension=5, replicas=2, timeout=-1)
        self.enable_teardown = True

        # run test
        yield

        # teardown
        if self.enable_teardown:
            self.pinecone.delete_index(name=self.index_name)

    # region: error-handling

    def test_configure_index_name_invalid(self):
        with pytest.raises(pinecone.core.client.exceptions.NotFoundException) as excinfo:
            self.pinecone.configure_index(name="nonexistant", replicas=2)

        exception_msg = str(excinfo.value)
        assert "Reason: Not Found" in exception_msg
        assert "HTTP response body: 404: Not Found" in exception_msg

    def test_configure_index_exceeds_quota(self):
        with pytest.raises(pinecone.core.client.exceptions.ApiException) as excinfo:
            self.pinecone.configure_index(name=self.index_name, replicas=10)

        exception_msg = str(excinfo.value)
        assert "Reason: Bad Request" in exception_msg
        assert "HTTP response body: The index exceeds the project quota of 5 pods by 5 pods." in exception_msg

    # endregion error-handling

    def test_configure_index_scale_replicas(self):
        index = self.pinecone.describe_index(name=self.index_name)
        assert index.replicas == 2

        # scale up
        self.pinecone.configure_index(name=self.index_name, replicas=3)
        index = self.pinecone.describe_index(name=self.index_name)
        assert index.replicas == 3

        # scale down
        self.pinecone.configure_index(name=self.index_name, replicas=1)
        index = self.pinecone.describe_index(name=self.index_name)
        assert index.replicas == 1

    def test_configure_index_scale_pod_type(self):
        index = self.pinecone.describe_index(name=self.index_name)
        assert index.pod_type == "p1.x1"

        # scale up
        self.pinecone.configure_index(name=self.index_name, pod_type="p1.x2")
        index = self.pinecone.describe_index(name=self.index_name)
        assert index.pod_type == "p1.x2"

        # scale down (expected error)
        with pytest.raises(pinecone.core.client.exceptions.ApiException) as excinfo:
            self.pinecone.configure_index(name=self.index_name, pod_type="p1.x1")

        exception_msg = str(excinfo.value)
        assert "Reason: Bad Request" in exception_msg
        assert "HTTP response body: scaling down pod type is not supported" in exception_msg
