import pytest
from pinecone import PineconeAsyncio, EmbedModel, CloudProvider, AwsRegion, IndexEmbed, Metric


@pytest.mark.asyncio
class TestCreateIndexForModel:
    @pytest.mark.parametrize(
        "model_val,cloud_val,region_val",
        [
            ("multilingual-e5-large", "aws", "us-east-1"),
            (EmbedModel.Multilingual_E5_Large, CloudProvider.AWS, AwsRegion.US_EAST_1),
            (EmbedModel.Pinecone_Sparse_English_V0, CloudProvider.AWS, AwsRegion.US_EAST_1),
        ],
    )
    async def test_create_index_for_model(self, model_val, index_name, cloud_val, region_val):
        pc = PineconeAsyncio()

        field_map = {"text": "my-sample-text"}
        index = await pc.create_index_for_model(
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
        await pc.close()

    async def test_create_index_for_model_with_index_embed_obj(self, index_name):
        pc = PineconeAsyncio()

        field_map = {"text": "my-sample-text"}
        index = await pc.create_index_for_model(
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
        await pc.close()

    @pytest.mark.parametrize(
        "model_val,metric_val",
        [(EmbedModel.Multilingual_E5_Large, Metric.COSINE), ("multilingual-e5-large", "cosine")],
    )
    async def test_create_index_for_model_with_index_embed_dict(
        self, index_name, model_val, metric_val
    ):
        pc = PineconeAsyncio()

        field_map = {"text": "my-sample-text"}
        index = await pc.create_index_for_model(
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
        await pc.close()
