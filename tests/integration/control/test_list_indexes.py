import pytest
import pinecone
import os
import tests.test_helpers as test_helpers


class TestListIndexes:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # setup
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        pinecone.init(api_key=api_key, environment=environment)

        self.index_name = test_helpers.generate_index_name()
        self.pinecone = pinecone
        self.pinecone.create_index(name=self.index_name, dimension=5, timeout=-1)

        # run test
        yield

        # teardown
        pinecone.delete_index(name=self.index_name, timeout=-1)

    def test_list_indexes(self):
        indexes = self.pinecone.list_indexes()

        assert indexes is not None
        assert len(indexes) > 0
        assert self.index_name in indexes
