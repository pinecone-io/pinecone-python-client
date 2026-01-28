"""Tests for deployment model classes."""

from pinecone import ServerlessDeployment, ByocDeployment, PodDeployment


class TestServerlessDeployment:
    def test_required_params(self):
        deployment = ServerlessDeployment(cloud="aws", region="us-east-1")
        assert deployment.cloud == "aws"
        assert deployment.region == "us-east-1"

    def test_to_dict(self):
        deployment = ServerlessDeployment(cloud="aws", region="us-east-1")
        result = deployment.to_dict()
        assert result == {"deployment_type": "serverless", "cloud": "aws", "region": "us-east-1"}

    def test_to_dict_gcp(self):
        deployment = ServerlessDeployment(cloud="gcp", region="us-central1")
        result = deployment.to_dict()
        assert result == {"deployment_type": "serverless", "cloud": "gcp", "region": "us-central1"}

    def test_to_dict_azure(self):
        deployment = ServerlessDeployment(cloud="azure", region="eastus")
        result = deployment.to_dict()
        assert result == {"deployment_type": "serverless", "cloud": "azure", "region": "eastus"}


class TestByocDeployment:
    def test_required_params(self):
        deployment = ByocDeployment(environment="aws-us-east-1-b92")
        assert deployment.environment == "aws-us-east-1-b92"

    def test_to_dict(self):
        deployment = ByocDeployment(environment="aws-us-east-1-b92")
        result = deployment.to_dict()
        assert result == {"deployment_type": "byoc", "environment": "aws-us-east-1-b92"}

    def test_to_dict_different_environment(self):
        deployment = ByocDeployment(environment="gcp-us-central1-abc")
        result = deployment.to_dict()
        assert result == {"deployment_type": "byoc", "environment": "gcp-us-central1-abc"}


class TestPodDeployment:
    def test_required_params(self):
        deployment = PodDeployment(environment="us-east-1-aws", pod_type="p1.x1")
        assert deployment.environment == "us-east-1-aws"
        assert deployment.pod_type == "p1.x1"
        assert deployment.replicas == 1
        assert deployment.shards == 1
        assert deployment.pods is None

    def test_to_dict_minimal(self):
        deployment = PodDeployment(environment="us-east-1-aws", pod_type="p1.x1")
        result = deployment.to_dict()
        assert result == {
            "deployment_type": "pod",
            "environment": "us-east-1-aws",
            "pod_type": "p1.x1",
            "replicas": 1,
            "shards": 1,
        }

    def test_to_dict_with_replicas(self):
        deployment = PodDeployment(environment="us-east-1-aws", pod_type="p1.x1", replicas=3)
        result = deployment.to_dict()
        assert result == {
            "deployment_type": "pod",
            "environment": "us-east-1-aws",
            "pod_type": "p1.x1",
            "replicas": 3,
            "shards": 1,
        }

    def test_to_dict_with_shards(self):
        deployment = PodDeployment(environment="us-east-1-aws", pod_type="p1.x1", shards=2)
        result = deployment.to_dict()
        assert result == {
            "deployment_type": "pod",
            "environment": "us-east-1-aws",
            "pod_type": "p1.x1",
            "replicas": 1,
            "shards": 2,
        }

    def test_to_dict_with_pods(self):
        deployment = PodDeployment(
            environment="us-east-1-aws", pod_type="p1.x1", replicas=2, shards=2, pods=4
        )
        result = deployment.to_dict()
        assert result == {
            "deployment_type": "pod",
            "environment": "us-east-1-aws",
            "pod_type": "p1.x1",
            "replicas": 2,
            "shards": 2,
            "pods": 4,
        }

    def test_to_dict_different_pod_types(self):
        for pod_type in ["s1.x1", "s1.x2", "p1.x2", "p2.x1"]:
            deployment = PodDeployment(environment="us-east-1-aws", pod_type=pod_type)
            result = deployment.to_dict()
            assert result["pod_type"] == pod_type


class TestDeploymentUsageExamples:
    """Test the usage examples from the ticket."""

    def test_serverless_deployment_example(self):
        deployment = ServerlessDeployment(cloud="gcp", region="us-central1")
        result = deployment.to_dict()
        assert result["deployment_type"] == "serverless"
        assert result["cloud"] == "gcp"
        assert result["region"] == "us-central1"

    def test_byoc_deployment_example(self):
        deployment = ByocDeployment(environment="aws-us-east-1-b92")
        result = deployment.to_dict()
        assert result["deployment_type"] == "byoc"
        assert result["environment"] == "aws-us-east-1-b92"
