import pytest
from pinecone.exceptions import PineconeApiException


class TestSparseIndex:
    def test_create_sparse_index_with_metric(self, client, create_sl_index_params):
        create_sl_index_params["metric"] = "dotproduct"

        create_sl_index_params["vector_type"] = "sparse"
        del create_sl_index_params["dimension"]

        client.create_index(**create_sl_index_params)
        desc = client.describe_index(create_sl_index_params["name"])
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"
        assert desc.dimension is None
        assert desc.deletion_protection == "disabled"  # default value

        for i in client.list_indexes():
            if i.name == create_sl_index_params["name"]:
                assert i.metric == "dotproduct"
                assert i.vector_type == "sparse"
                assert i.dimension is None
                assert i.deletion_protection == "disabled"
                break
            else:
                assert i.vector_type is not None

    def test_sparse_index_deletion_protection(self, client, create_sl_index_params):
        create_sl_index_params["metric"] = "dotproduct"
        create_sl_index_params["vector_type"] = "sparse"
        create_sl_index_params["deletion_protection"] = "enabled"
        del create_sl_index_params["dimension"]

        client.create_index(**create_sl_index_params)

        desc = client.describe_index(create_sl_index_params["name"])
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"
        assert desc.dimension is None
        assert desc.deletion_protection == "enabled"

        with pytest.raises(PineconeApiException) as e:
            client.delete_index(create_sl_index_params["name"], -1)
        assert "Deletion protection is enabled for this index" in str(e.value)

        client.configure_index(create_sl_index_params["name"], deletion_protection="disabled")

        desc2 = client.describe_index(create_sl_index_params["name"])
        assert desc2.deletion_protection == "disabled"

        client.delete_index(create_sl_index_params["name"], -1)


class TestSparseIndexErrorCases:
    def test_exception_when_passing_dimension(self, client, create_sl_index_params):
        create_sl_index_params["metric"] = "dotproduct"
        create_sl_index_params["dimension"] = 10
        create_sl_index_params["vector_type"] = "sparse"

        with pytest.raises(ValueError) as e:
            client.create_index(**create_sl_index_params)
        assert "dimension should not be specified for sparse indexes" in str(e.value)

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_sparse_only_supports_dotproduct(self, client, create_sl_index_params, metric):
        create_sl_index_params["metric"] = metric
        create_sl_index_params["vector_type"] = "sparse"
        del create_sl_index_params["dimension"]

        with pytest.raises(PineconeApiException) as e:
            client.create_index(**create_sl_index_params)
        assert "Sparse vector indexes must use the metric dotproduct" in str(e.value)
