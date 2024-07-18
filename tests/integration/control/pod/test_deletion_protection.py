import pytest
from pinecone import PodSpec


class TestDeletionProtection:
    def test_deletion_protection(self, client, index_name, environment):
        client.create_index(
            name=index_name, dimension=2, deletion_protection="enabled", spec=PodSpec(environment=environment)
        )
        desc = client.describe_index(index_name)
        print(desc.deletion_protection)
        print(desc.deletion_protection.__class__)
        assert desc.deletion_protection == "enabled"

        with pytest.raises(Exception) as e:
            client.delete_index(index_name)
        assert "Deletion protection is enabled for this index" in str(e.value)

        client.configure_index(index_name, deletion_protection="disabled")
        desc = client.describe_index(index_name)
        assert desc.deletion_protection == "disabled"

        client.delete_index(index_name)

    def test_configure_index_with_deletion_protection(self, client, index_name, environment):
        client.create_index(
            name=index_name, dimension=2, deletion_protection="enabled", spec=PodSpec(environment=environment)
        )
        desc = client.describe_index(index_name)
        assert desc.deletion_protection == "enabled"

        # Changing replicas only should not change deletion protection
        client.configure_index(name=index_name, replicas=2)
        desc = client.describe_index(index_name)
        assert desc.spec.pod.replicas == 2
        assert desc.deletion_protection == "enabled"

        # Changing both replicas and delete protection in one shot
        client.configure_index(name=index_name, replicas=3, deletion_protection="disabled")
        desc = client.describe_index(index_name)
        assert desc.spec.pod.replicas == 3
        assert desc.deletion_protection == "disabled"

        # Cleanup
        client.delete_index(index_name)
