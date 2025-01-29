import pytest
from pinecone import EmbedModel, CloudProvider, AwsRegion, IndexEmbed, Metric


class TestCreateIndexForModel:
    @pytest.mark.parametrize(
        "model_val,cloud_val,region_val",
        [
            ("multilingual-e5-large", "aws", "us-east-1"),
            (EmbedModel.Multilingual_E5_Large, CloudProvider.AWS, AwsRegion.US_EAST_1),
            (EmbedModel.Pinecone_Sparse_English_V0, CloudProvider.AWS, AwsRegion.US_EAST_1),
        ],
    )
    def test_create_index_for_model(self, client, model_val, index_name, cloud_val, region_val):
        field_map = {"text": "my-sample-text"}
        index = client.create_index_for_model(
            name=index_name,
            cloud=cloud_val,
            region=region_val,
            embed={"model": model_val, "field_map": field_map},
            timeout=-1,
        )
        assert index.name == index_name
        assert index.spec.serverless.cloud == "aws"
        assert index.spec.serverless.region == "us-east-1"
        assert index.embed.field_map == field_map
        if isinstance(model_val, EmbedModel):
            assert index.embed.model == model_val.value
        else:
            assert index.embed.model == model_val

    def test_create_index_for_model_with_index_embed_obj(self, client, index_name):
        field_map = {"text": "my-sample-text"}
        index = client.create_index_for_model(
            name=index_name,
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_EAST_1,
            embed=IndexEmbed(
                metric=Metric.COSINE, model=EmbedModel.Multilingual_E5_Large, field_map=field_map
            ),
            timeout=-1,
        )
        assert index.name == index_name
        assert index.spec.serverless.cloud == "aws"
        assert index.spec.serverless.region == "us-east-1"
        assert index.embed.field_map == field_map
        assert index.embed.model == EmbedModel.Multilingual_E5_Large.value

    @pytest.mark.parametrize(
        "model_val,metric_val",
        [(EmbedModel.Multilingual_E5_Large, Metric.COSINE), ("multilingual-e5-large", "cosine")],
    )
    def test_create_index_for_model_with_index_embed_dict(
        self, client, index_name, model_val, metric_val
    ):
        field_map = {"text": "my-sample-text"}
        index = client.create_index_for_model(
            name=index_name,
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_EAST_1,
            embed={"metric": metric_val, "field_map": field_map, "model": model_val},
            timeout=-1,
        )
        assert index.name == index_name
        assert index.spec.serverless.cloud == "aws"
        assert index.spec.serverless.region == "us-east-1"
        assert index.embed.field_map == field_map
        assert index.embed.model == EmbedModel.Multilingual_E5_Large.value
