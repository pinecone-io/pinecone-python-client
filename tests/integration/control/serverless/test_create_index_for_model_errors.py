import pytest
from pinecone import (
    EmbedModel,
    CloudProvider,
    AwsRegion,
    Metric,
    PineconeApiException,
    PineconeApiValueError,
)


class TestCreateIndexForModelErrors:
    def test_create_index_for_model_with_invalid_model(self, client, index_name):
        with pytest.raises(PineconeApiException) as e:
            client.create_index_for_model(
                name=index_name,
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
                embed={
                    "model": "invalid-model",
                    "field_map": {"text": "my-sample-text"},
                    "metric": Metric.COSINE,
                },
                timeout=-1,
            )
        assert "Model invalid-model not found." in str(e.value)

    def test_invalid_cloud(self, client, index_name):
        with pytest.raises(PineconeApiValueError) as e:
            client.create_index_for_model(
                name=index_name,
                cloud="invalid-cloud",
                region=AwsRegion.US_EAST_1,
                embed={
                    "model": EmbedModel.Multilingual_E5_Large,
                    "field_map": {"text": "my-sample-text"},
                    "metric": Metric.COSINE,
                },
                timeout=-1,
            )
        assert "Invalid value for `cloud`" in str(e.value)

    def test_invalid_region(self, client, index_name):
        with pytest.raises(PineconeApiException) as e:
            client.create_index_for_model(
                name=index_name,
                cloud=CloudProvider.AWS,
                region="invalid-region",
                embed={
                    "model": EmbedModel.Multilingual_E5_Large,
                    "field_map": {"text": "my-sample-text"},
                    "metric": Metric.COSINE,
                },
                timeout=-1,
            )
        assert "invalid-region not found" in str(e.value)

    def test_create_index_for_model_with_invalid_field_map(self, client, index_name):
        with pytest.raises(PineconeApiException) as e:
            client.create_index_for_model(
                name=index_name,
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
                embed={
                    "model": EmbedModel.Multilingual_E5_Large,
                    "field_map": {"invalid_field": "my-sample-text"},
                    "metric": Metric.COSINE,
                },
                timeout=-1,
            )
        assert "Missing required key 'text'" in str(e.value)

    def test_create_index_for_model_with_invalid_metric(self, client, index_name):
        with pytest.raises(PineconeApiValueError) as e:
            client.create_index_for_model(
                name=index_name,
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
                embed={
                    "model": EmbedModel.Multilingual_E5_Large,
                    "field_map": {"text": "my-sample-text"},
                    "metric": "invalid-metric",
                },
                timeout=-1,
            )
        assert "Invalid value for `metric`" in str(e.value)

    def test_create_index_for_model_with_missing_name(self, client):
        with pytest.raises(TypeError) as e:
            client.create_index_for_model(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
                embed={
                    "model": EmbedModel.Multilingual_E5_Large,
                    "field_map": {"text": "my-sample-text"},
                    "metric": Metric.EUCLIDEAN,
                },
                timeout=-1,
            )
        assert "name" in str(e.value)

    def test_create_index_with_missing_model(self, client, index_name):
        with pytest.raises(ValueError) as e:
            client.create_index_for_model(
                name=index_name,
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
                embed={"field_map": {"text": "my-sample-text"}, "metric": Metric.COSINE},
                timeout=-1,
            )
        assert "model is required" in str(e.value)

    def test_create_index_with_missing_field_map(self, client, index_name):
        with pytest.raises(ValueError) as e:
            client.create_index_for_model(
                name=index_name,
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
                embed={"model": EmbedModel.Multilingual_E5_Large, "metric": Metric.COSINE},
                timeout=-1,
            )
        assert "field_map is required" in str(e.value)
