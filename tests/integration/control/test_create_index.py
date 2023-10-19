import pytest
import pinecone
import os
import tests.test_helpers as test_helpers


class TestCreateIndex:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # setup
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")

        self.index_name = test_helpers.generate_index_name()
        self.pinecone = pinecone
        self.pinecone.init(api_key=api_key, environment=environment)
        self.enable_teardown = True

        # run test
        yield

        # teardown
        if self.enable_teardown:
            pinecone.delete_index(name=self.index_name, timeout=-1)

    # region: error handling

    def test_create_index_invalid_name(self):
        self.enable_teardown = False

        with pytest.raises(pinecone.core.client.exceptions.ApiException) as excinfo:
            self.pinecone.create_index(name=self.index_name + "-", dimension=5)

        exception_msg = str(excinfo.value)
        assert "Reason: Bad Request" in exception_msg
        assert "must be an empty string or consist of alphanumeric characters" in exception_msg

    def test_create_index_nonexistant_collection(self):
        self.enable_teardown = False

        with pytest.raises(pinecone.core.client.exceptions.ApiException) as excinfo:
            self.pinecone.create_index(name=self.index_name, dimension=5, source_collection="nonexistant")

        exception_msg = str(excinfo.value)
        assert "Reason: Bad Request" in exception_msg
        assert "HTTP response body: failed to fetch source collection nonexistant" in exception_msg

    def test_create_index_insufficient_quota(self):
        self.enable_teardown = False

        with pytest.raises(pinecone.core.client.exceptions.ApiException) as excinfo:
            self.pinecone.create_index(name=self.index_name, dimension=5, replicas=20)

        exception_msg = str(excinfo.value)
        assert "Reason: Bad Request" in exception_msg
        assert "HTTP response body: The index exceeds the project quota of 5 pods by 15 pods." in exception_msg

    # endregion

    def test_create_index(self):
        self.pinecone.create_index(name=self.index_name, dimension=10, timeout=-1)
        index = self.pinecone.describe_index(name=self.index_name)

        assert index.name == self.index_name
        assert index.dimension == 10
        assert index.metric == "cosine"
        assert index.pods == 1
        assert index.replicas == 1
        assert index.shards == 1

    def test_create_index_with_options(self):
        self.pinecone.create_index(
            name=self.index_name, dimension=10, metric="euclidean", replicas=2, pod_type="p1.x2", timeout=-1
        )
        index = self.pinecone.describe_index(name=self.index_name)

        assert index.name == self.index_name
        assert index.dimension == 10
        assert index.metric == "euclidean"
        assert index.pods == 2
        assert index.replicas == 2
        assert index.shards == 1
