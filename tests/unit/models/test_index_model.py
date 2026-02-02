from pinecone.db_control.models import IndexModel
from pinecone import CloudProvider, AwsRegion

from tests.fixtures import make_index_model, make_index_status


class TestIndexModel:
    def test_index_model(self):
        openapi_model = make_index_model(
            name="test-index-1",
            dimension=2,
            metric="cosine",
            host="https://test-index-1.pinecone.io",
            status=make_index_status(ready=True, state="Ready"),
            deletion_protection="enabled",
            spec={
                "serverless": {
                    "cloud": CloudProvider.AWS.value,
                    "region": AwsRegion.US_EAST_1.value,
                }
            },
        )

        wrapped = IndexModel(openapi_model)

        assert wrapped.name == "test-index-1"
        assert wrapped.dimension == 2
        assert wrapped.metric == "cosine"
        assert wrapped.host == "https://test-index-1.pinecone.io"
        assert wrapped.status.ready == True
        assert wrapped.status.state == "Ready"
        assert wrapped.deletion_protection == "enabled"

        assert wrapped["name"] == "test-index-1"
