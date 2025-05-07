import pytest
import time
from pinecone import (
    Pinecone,
    Metric,
    VectorType,
    DeletionProtection,
    ServerlessSpec,
    PodSpec,
    CloudProvider,
    AwsRegion,
    PineconeApiValueError,
    PineconeApiException,
    PineconeApiTypeError,
    PodIndexEnvironment,
)


class TestCreateServerlessIndexHappyPath:
    def test_create_index(self, pc: Pinecone, index_name):
        resp = pc.db.index.create(
            name=index_name,
            dimension=10,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
        )
        assert resp.name == index_name
        assert resp.dimension == 10
        assert resp.metric == "cosine"  # default value
        assert resp.vector_type == "dense"  # default value
        assert resp.deletion_protection == "disabled"  # default value

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 10
        assert desc.metric == "cosine"
        assert desc.deletion_protection == "disabled"  # default value
        assert desc.vector_type == "dense"  # default value

    def test_create_skip_wait(self, pc, index_name):
        resp = pc.db.index.create(
            name=index_name,
            dimension=10,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            timeout=-1,
        )
        assert resp.name == index_name
        assert resp.dimension == 10
        assert resp.metric == "cosine"

    def test_create_infinite_wait(self, pc, index_name):
        resp = pc.db.index.create(
            name=index_name,
            dimension=10,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            timeout=None,
        )
        assert resp.name == index_name
        assert resp.dimension == 10
        assert resp.metric == "cosine"

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dotproduct"])
    def test_create_default_index_with_metric(self, pc, create_sl_index_params, metric):
        create_sl_index_params["metric"] = metric
        pc.db.index.create(**create_sl_index_params)
        desc = pc.db.index.describe(create_sl_index_params["name"])
        if isinstance(metric, str):
            assert desc.metric == metric
        else:
            assert desc.metric == metric.value
        assert desc.vector_type == "dense"

    @pytest.mark.parametrize(
        "metric_enum,vector_type_enum,dim,tags",
        [
            (Metric.COSINE, VectorType.DENSE, 10, None),
            (Metric.EUCLIDEAN, VectorType.DENSE, 10, {"env": "prod"}),
            (Metric.DOTPRODUCT, VectorType.SPARSE, None, {"env": "dev"}),
        ],
    )
    def test_create_with_enum_values(
        self, pc, index_name, metric_enum, vector_type_enum, dim, tags
    ):
        args = {
            "name": index_name,
            "metric": metric_enum,
            "vector_type": vector_type_enum,
            "deletion_protection": DeletionProtection.DISABLED,
            "spec": ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            "tags": tags,
        }
        if dim is not None:
            args["dimension"] = dim

        pc.db.index.create(**args)

        desc = pc.db.index.describe(index_name)
        assert desc.metric == metric_enum.value
        assert desc.vector_type == vector_type_enum.value
        assert desc.dimension == dim
        assert desc.deletion_protection == DeletionProtection.DISABLED.value
        assert desc.name == index_name
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"
        if tags:
            assert desc.tags.to_dict() == tags

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dotproduct"])
    def test_create_dense_index_with_metric(self, pc, create_sl_index_params, metric):
        create_sl_index_params["metric"] = metric
        create_sl_index_params["vector_type"] = VectorType.DENSE
        pc.db.index.create(**create_sl_index_params)
        desc = pc.db.index.describe(create_sl_index_params["name"])
        assert desc.metric == metric
        assert desc.vector_type == "dense"

    def test_create_with_optional_tags(self, pc, create_sl_index_params):
        tags = {"foo": "FOO", "bar": "BAR"}
        create_sl_index_params["tags"] = tags
        pc.db.index.create(**create_sl_index_params)
        desc = pc.db.index.describe(create_sl_index_params["name"])
        assert desc.tags.to_dict() == tags


class TestCreatePodIndexHappyPath:
    def test_create_index_minimal_config(
        self, pc: Pinecone, index_name, pod_environment, index_tags
    ):
        pc.db.index.create(
            name=index_name,
            dimension=10,
            metric="cosine",
            spec=PodSpec(environment=pod_environment),
            tags=index_tags,
        )

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 10
        assert desc.metric == "cosine"
        assert desc.spec.pod.environment == pod_environment
        assert desc.tags.to_dict() == index_tags
        assert desc.status.ready == True
        assert desc.status.state == "Ready"
        assert desc.vector_type == "dense"

    def test_create_index_with_spec_options(
        self, pc: Pinecone, index_name, pod_environment, index_tags
    ):
        pc.db.index.create(
            name=index_name,
            dimension=10,
            metric="cosine",
            spec=PodSpec(
                environment=pod_environment,
                pod_type="p1.x2",
                replicas=2,
                metadata_config={"indexed": ["foo", "bar"]},
            ),
            tags=index_tags,
        )

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.dimension == 10
        assert desc.metric == "cosine"
        assert desc.spec.pod.environment == pod_environment
        assert desc.spec.pod.pod_type == "p1.x2"
        assert desc.spec.pod.replicas == 2
        assert desc.spec.pod.metadata_config.indexed == ["foo", "bar"]

    def test_create_index_with_deletion_protection(
        self, pc: Pinecone, index_name, pod_environment, index_tags
    ):
        pc.db.index.create(
            name=index_name,
            dimension=10,
            metric="cosine",
            spec=PodSpec(environment=pod_environment),
            tags=index_tags,
            deletion_protection=DeletionProtection.ENABLED,
        )

        try:
            pc.db.index.delete(name=index_name)
        except PineconeApiException as e:
            assert "Deletion protection is enabled for this index" in str(e)

        pc.db.index.configure(name=index_name, deletion_protection=DeletionProtection.DISABLED)
        max_wait_time = 60
        while pc.db.index.describe(name=index_name).status.ready == False:
            time.sleep(1)
            max_wait_time -= 1
            if max_wait_time <= 0:
                raise Exception("Index did not become ready in time")

        pc.db.index.delete(name=index_name)
        assert pc.db.index.has(name=index_name) == False


class TestCreatePodIndexApiErrorCases:
    def test_pod_index_does_not_support_sparse_vectors(self, pc, index_name, index_tags):
        with pytest.raises(PineconeApiException) as e:
            pc.db.index.create(
                name=index_name,
                metric="dotproduct",
                spec=PodSpec(environment=PodIndexEnvironment.US_EAST1_AWS),
                vector_type="sparse",
                tags=index_tags,
            )
        assert "Sparse vector type is not supported for pod indexes" in str(e.value)


class TestCreateServerlessIndexApiErrorCases:
    def test_create_index_with_invalid_name(self, pc, create_sl_index_params):
        create_sl_index_params["name"] = "Invalid-name"
        with pytest.raises(PineconeApiException):
            pc.db.index.create(**create_sl_index_params)

    def test_create_index_invalid_metric(self, pc, create_sl_index_params):
        create_sl_index_params["metric"] = "invalid"
        with pytest.raises(PineconeApiValueError):
            pc.db.index.create(**create_sl_index_params)

    def test_create_index_with_invalid_neg_dimension(self, pc, create_sl_index_params):
        create_sl_index_params["dimension"] = -1
        with pytest.raises(PineconeApiValueError):
            pc.db.index.create(**create_sl_index_params)

    def test_create_index_that_already_exists(self, pc, create_sl_index_params):
        pc.db.index.create(**create_sl_index_params)
        with pytest.raises(PineconeApiException):
            pc.db.index.create(**create_sl_index_params)


class TestCreateServerlessIndexWithTimeout:
    def test_create_index_default_timeout(self, pc, create_sl_index_params):
        create_sl_index_params["timeout"] = None
        pc.db.index.create(**create_sl_index_params)
        # Waits infinitely for index to be ready
        desc = pc.db.index.describe(create_sl_index_params["name"])
        assert desc.status.ready == True

    def test_create_index_when_timeout_set(self, pc, create_sl_index_params):
        create_sl_index_params["timeout"] = (
            1000  # effectively infinite, but different code path from None
        )
        pc.db.index.create(**create_sl_index_params)
        desc = pc.db.index.describe(name=create_sl_index_params["name"])
        assert desc.status.ready == True

    def test_create_index_with_negative_timeout(self, pc, create_sl_index_params):
        create_sl_index_params["timeout"] = -1
        pc.db.index.create(**create_sl_index_params)
        desc = pc.db.index.describe(create_sl_index_params["name"])
        # Returns immediately without waiting for index to be ready
        assert desc.status.ready in [False, True]


class TestCreateIndexTypeErrorCases:
    def test_create_index_with_invalid_str_dimension(self, pc, create_sl_index_params):
        create_sl_index_params["dimension"] = "10"
        with pytest.raises(PineconeApiTypeError):
            pc.db.index.create(**create_sl_index_params)

    def test_create_index_with_missing_dimension(self, pc, create_sl_index_params):
        del create_sl_index_params["dimension"]
        with pytest.raises(PineconeApiException):
            pc.db.index.create(**create_sl_index_params)

    def test_create_index_w_incompatible_options(self, pc, create_sl_index_params):
        create_sl_index_params["pod_type"] = "p1.x2"
        create_sl_index_params["environment"] = "us-east1-gcp"
        create_sl_index_params["replicas"] = 2
        with pytest.raises(TypeError):
            pc.db.index.create(**create_sl_index_params)

    @pytest.mark.parametrize("required_option", ["name", "spec", "dimension"])
    def test_create_with_missing_required_options(
        self, pc, create_sl_index_params, required_option
    ):
        del create_sl_index_params[required_option]
        with pytest.raises(Exception) as e:
            pc.db.index.create(**create_sl_index_params)
        assert required_option.lower() in str(e.value).lower()


class TestSparseIndex:
    def test_create_sparse_index_minimal_config(self, pc: Pinecone, index_name, index_tags):
        pc.db.index.create(
            name=index_name,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
            vector_type=VectorType.SPARSE,
            tags=index_tags,
        )

        desc = pc.db.index.describe(name=index_name)
        assert desc.name == index_name
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"


class TestSparseIndexErrorCases:
    def test_sending_dimension_with_sparse_index(self, pc, index_tags):
        with pytest.raises(ValueError) as e:
            pc.db.index.create(
                name="test-index",
                dimension=10,
                metric="dotproduct",
                vector_type=VectorType.SPARSE,
                spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
                tags=index_tags,
            )
        assert "dimension should not be specified for sparse indexes" in str(e.value)

    @pytest.mark.parametrize("bad_metric", ["cosine", "euclidean"])
    def test_sending_metric_other_than_dotproduct_with_sparse_index(
        self, pc, index_tags, bad_metric
    ):
        with pytest.raises(PineconeApiException) as e:
            pc.db.index.create(
                name="test-index",
                metric=bad_metric,
                vector_type=VectorType.SPARSE,
                spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
                tags=index_tags,
            )
        assert "Sparse vector indexes must use the metric dotproduct" in str(e.value)
