import pytest
import pinecone
import os
import tests.test_helpers as test_helpers


class TestDescribeIndex:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # setup
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")

        self.index_name = test_helpers.generate_index_name()
        self.pinecone = pinecone
        self.pinecone.init(api_key=api_key, environment=environment)
        self.pinecone.create_index(name=self.index_name, dimension=5, timeout=-1)
        self.enable_teardown = True

        # run test
        yield

        # teardown
        if self.enable_teardown:
            self.pinecone.delete_index(name=self.index_name, timeout=-1)

    def test_describe_index(self):
        description = self.pinecone.describe_index(name=self.index_name)
        assert description.name == self.index_name
        assert description.dimension == 5
        assert description.metric == "cosine"
        assert description.pods == 1
        assert description.replicas == 1
        assert description.shards == 1

    def test_describe_index_invalid_name(self):
        with pytest.raises(pinecone.core.client.exceptions.NotFoundException) as excinfo:
            self.pinecone.describe_index(name="invalid-index-name")

        exception_msg = str(excinfo.value)
        assert "Reason: Not Found" in exception_msg
        assert "HTTP response body: 404: Not Found" in exception_msg
