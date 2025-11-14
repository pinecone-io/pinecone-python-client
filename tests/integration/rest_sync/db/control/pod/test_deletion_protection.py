import pytest
import time
from pinecone import PodSpec


@pytest.mark.skip(reason="slow")
class TestDeletionProtection:
    def test_deletion_protection(self, client, index_name, environment):
        client.create_index(
            name=index_name,
            dimension=2,
            deletion_protection="enabled",
            spec=PodSpec(environment=environment),
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
            name=index_name,
            dimension=2,
            deletion_protection="enabled",
            spec=PodSpec(environment=environment),
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

        # Wait up to 30*2 seconds for the index to be ready before attempting to delete
        for t in range(1, 30):
            delta = 2
            desc = client.describe_index(index_name)
            if desc.status.state == "Ready":
                print(f"Index {index_name} is ready after {(t - 1) * delta} seconds")
                break
            print("Index is not ready yet. Waiting for 2 seconds.")
            time.sleep(delta)

        attempts = 0
        while attempts < 12:
            try:
                client.delete_index(index_name)
                break
            except Exception as e:
                attempts += 1
                print(f"Failed to delete index {index_name} on attempt {attempts}.")
                print(f"Error: {e}")
                client.describe_index(index_name)
                time.sleep(10)
