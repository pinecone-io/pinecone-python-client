from pinecone import ByocSpec, ServerlessSpec
from pinecone.db_control.request_factory import PineconeDBControlRequestFactory


class TestIndexRequestFactory:
    def test_create_index_request_with_spec_byoc(self):
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec=ByocSpec(environment="test-byoc-spec-id"),
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.byoc.environment == "test-byoc-spec-id"
        assert req.vector_type == "dense"
        assert req.deletion_protection.value == "disabled"

    def test_create_index_request_with_spec_serverless(self):
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.serverless.cloud == "aws"
        assert req.spec.serverless.region == "us-east-1"
        assert req.vector_type == "dense"
        assert req.deletion_protection.value == "disabled"

    def test_create_index_request_with_spec_serverless_dict(self):
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.serverless.cloud == "aws"
        assert req.spec.serverless.region == "us-east-1"
        assert req.vector_type == "dense"
        assert req.deletion_protection.value == "disabled"

    def test_create_index_request_with_spec_byoc_dict(self):
        req = PineconeDBControlRequestFactory.create_index_request(
            name="test-index",
            metric="cosine",
            dimension=1024,
            spec={"byoc": {"environment": "test-byoc-spec-id"}},
        )
        assert req.name == "test-index"
        assert req.metric == "cosine"
        assert req.dimension == 1024
        assert req.spec.byoc.environment == "test-byoc-spec-id"
        assert req.vector_type == "dense"
        assert req.deletion_protection.value == "disabled"
