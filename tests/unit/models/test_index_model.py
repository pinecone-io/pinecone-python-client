from pinecone.core.openapi.db_control.models import (
    IndexModel as OpenApiIndexModel,
    IndexModelStatus,
    IndexModelSpec,
    ServerlessSpec,
    DeletionProtection,
)
from pinecone.db_control.models import IndexModel
from pinecone import CloudProvider, AwsRegion


class TestIndexModel:
    def test_index_model(self):
        openapi_model = OpenApiIndexModel(
            name="test-index-1",
            dimension=2,
            metric="cosine",
            host="https://test-index-1.pinecone.io",
            status=IndexModelStatus(ready=True, state="Ready"),
            deletion_protection=DeletionProtection("enabled"),
            spec=IndexModelSpec(
                serverless=ServerlessSpec(
                    cloud=CloudProvider.AWS.value, region=AwsRegion.US_EAST_1.value
                )
            ),
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
